"""Audit speed factors, paired inputs, saved signals and independent dynamics."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from pelican_e2.data import state_features, advance
from pelican_e2.model import library
from pelican_e3.plant import derivative
from run_pelican_e2 import checksum, restore_model, write_json
from run_pelican_speed_sweep import read_json, check_preserved, SNAPSHOTS


def reference_position_velocity(t, cfg):
    """Vectorized independent expression for this experiment's eight reference."""
    w, r, ramp = (cfg[k] for k in ['trajectory_frequency_rad_s', 'trajectory_radius_m', 'trajectory_ramp_s'])
    s = np.minimum(t / ramp, 1)
    phase = np.where(t < ramp, w * ramp * (s**3 - .5 * s**4), w * (t - .5 * ramp))
    rate = w * (3 * s**2 - 2 * s**3)
    p = np.c_[r * np.sin(phase), .5 * r * np.sin(2 * phase), np.ones(len(t))]
    v = np.c_[r * np.cos(phase) * rate, r * np.cos(2 * phase) * rate, np.zeros(len(t))]
    return np.c_[p, v]


def verify(out):
    out = out.resolve()
    frozen, settings = read_json(out / 'FROZEN.json'), read_json(out / 'config.json')
    for path, value in frozen['source_sha256'].items():
        assert checksum(ROOT / path) == value, path
    for path, value in frozen['artifact_sha256'].items():
        assert checksum(out / path) == value, path
    source, stage = ROOT / settings['base_run'], settings[frozen['stage']]
    assert settings['controllers'] == read_json(source / 'config.json')['controllers']
    observer = read_json(out / 'observer.json')
    assert observer == read_json(source / 'config.json')['observer']
    for name in SNAPSHOTS:
        assert (out / name).read_bytes() == (source / name).read_bytes(), name
    base = read_json(out / 'base_e3_config.json')
    assert base == read_json(source / 'e3_config.json')
    assert not set(settings['development']['seeds']) & set(settings['formal']['seeds'])
    if frozen['development']:
        dev = frozen['development']
        assert checksum(ROOT / dev['path'] / 'verification.json') == dev['verification_sha256']
    params, motor = (read_json(out / name) for name in SNAPSHOTS[:2])
    models = {n: restore_model(m) for n, m in read_json(out / 'corrected_models.json').items()}
    plan = read_json(out / 'planned_cases.json')
    rows = pd.read_csv(out / 'tracking_per_run.csv').set_index('case_id')
    completed = read_json(out / 'COMPLETED.json')
    assert checksum(out / 'tracking_per_run.csv') == completed['tracking_sha256']
    expected = {(i, kind, seed, name) for i in range(len(settings['frequency_rad_s']))
                for kind in stage['trajectories'] for seed in stage['seeds'] for name in settings['controllers']}
    assert len(plan) == len(expected) == len(rows) == completed['cases']
    assert {(c['speed_index'], c['trajectory'], c['seed'], c['controller']) for c in plan} == expected
    assert rows.index.is_unique and set(rows.index) == {c['case_id'] for c in plan}
    alpha = 1 - np.exp(-base['dt'] / np.asarray(observer['time_constant_s']))
    limit, gate = np.asarray(observer['estimate_limit']), np.asarray(observer['residual_gate'])
    poles = np.asarray(motor['poles'])
    gain, offset = np.asarray(motor['coefficients']).T
    counts = dict(cases=0, completed_cases=0, observed_transitions=0, residual_rows=0,
                  independent_plant_steps=0, causal_bias_checks=0)
    max_residual, max_physical = 0., 0.
    initial_values = [np.load(out / 'initial_conditions' / f'{s}.npz')['truth'][0] for s in stage['seeds']]
    assert len(np.unique(initial_values, axis=0)) == len(stage['seeds'])
    for case in plan:
        cfg = read_json(out / 'reference_configs' / f"speed{case['speed_index']}.json")
        assert set(cfg) == set(base)
        for key in base:
            if key not in ['duration_s', 'trajectory_frequency_rad_s']:
                assert cfg[key] == base[key], key
        assert cfg['duration_s'] == stage['duration_s']
        assert cfg['trajectory_frequency_rad_s'] == case['frequency_rad_s'] == settings['frequency_rad_s'][case['speed_index']]
        folder = out / 'cases' / case['case_id']
        m = read_json(folder / 'metrics.json')
        for key, value in case.items():
            assert m[key] == value
        with np.load(folder / 'trajectory.npz') as f:
            d = {k: f[k] for k in f.files}
        with np.load(out / 'initial_conditions' / f"{case['seed']}.npz") as f:
            initial = {k: f[k] for k in f.files}
        frame = pd.read_csv(folder / 'control.csv')
        x, truth, u, t = d['observation'], d['physical'], d['command'], d['t']
        n, dt = len(u), cfg['dt']
        hold = round(cfg['control_dt'] / dt)
        assert len(t) == len(truth) == len(x) == n + 1
        assert_array_equal(truth[0], initial['truth'][-1])
        assert_array_equal(x[0], initial['states'][-1])
        assert_array_equal(d['initial_history_states'], initial['states'])
        assert_array_equal(d['initial_history_commands'], initial['commands'])
        assert_allclose(t, np.arange(n + 1) * dt, atol=1e-14)
        assert_allclose(x[1:, :3], truth[1:, :3], atol=1e-14)
        assert_allclose(x[1:, 3:6], np.diff(truth[:, :3], axis=0) / dt, atol=1e-12)
        assert_allclose(x[:, 6:10], truth[:, 6:10], atol=1e-14)
        rates = (Rotation.from_quat(truth[:-1, 6:10]).inv() * Rotation.from_quat(truth[1:, 6:10])).as_rotvec() / dt
        assert_allclose(x[1:, 10:13], rates, atol=1e-12)
        assert case['trajectory'] == 'figure8'
        assert_allclose(d['reference'][:, :6], reference_position_velocity(t, cfg), atol=1e-12)
        estimates = np.zeros((n + 1, 6))
        if case['controller'] != 'pd':
            model = models[case['controller']]
            n0 = len(initial['commands'])
            xa = np.vstack([initial['states'], x[1:]])
            ua = np.vstack([initial['commands'], u])
            if model['motor']:
                prior = gain * np.asarray(params['hover_command']) + offset
                inputs = []
                for command in ua:
                    prior = poles * prior + (1 - poles) * (gain * command + offset)
                    inputs.append(prior.copy())
                inputs = np.asarray(inputs)
                assert_allclose(d['motor_prior'], inputs[n0 - 1:], atol=1e-12)
            else:
                inputs = ua
            z = (np.c_[state_features(xa[:-1], cfg['gravity']), inputs] - model['mu']) / model['scale']
            idx = n0 + np.arange(n)
            phi = library(z[idx], {lag: z[idx - lag] for lag in model['lags']}, model['lags'], model['degree'])
            target = ((phi / model['feature_scale']) @ model['coefficients'].T) * model['y_scale'] + model['y_mu']
            predicted = advance(x[:-1], target, cfg)
            residual = np.c_[Rotation.from_quat(x[:-1, 6:10]).inv().apply((x[1:, 3:6] - predicted[:, 3:6]) / dt),
                             (x[1:, 10:13] - predicted[:, 10:13]) / dt]
            assert_allclose(residual, d['residual'], atol=1e-10, rtol=1e-10)
            max_residual = max(max_residual, float(np.max(np.abs(residual - d['residual']))))
            for k, raw in enumerate(residual):
                good = bool(np.isfinite(raw).all() and np.all(np.abs(raw) <= gate))
                candidate = estimates[k] + alpha * (raw - estimates[k]) if good else estimates[k]
                estimates[k + 1] = np.clip(candidate, -limit, limit)
                assert bool(d['accepted'][k]) == good
                assert bool(d['clipped'][k]) == bool(np.any(candidate != estimates[k + 1]))
            counts['residual_rows'] += n
            counts['causal_bias_checks'] += len(frame)
        assert_allclose(estimates, d['bias_estimate'], atol=1e-10)
        indices = frame.step.to_numpy(dtype=int)
        assert_array_equal(indices, np.arange(0, n, hold))
        assert_allclose(frame[[f'bias{j}' for j in range(6)]], estimates[indices], atol=1e-10)
        commands = frame[[f'u{j}' for j in range(1, 5)]].to_numpy()
        assert_allclose(np.repeat(commands, hold, axis=0)[:n], u, atol=1e-12)
        delta = np.diff(np.vstack([initial['commands'][-1], commands]), axis=0)
        assert_allclose(np.abs(delta).max(1), frame.max_command_change, atol=1e-12)
        lower, upper = np.asarray(params['command_lower']), np.asarray(params['command_upper'])
        bounds = int(np.any((u < lower - 1e-7) | (u > upper + 1e-7), axis=1).sum())
        slew = int(np.any(np.abs(delta) > cfg['command_delta_per_control'] + 1e-7, axis=1).sum())
        assert m['input_bound_violations'] == bounds and m['input_slew_violations'] == slew
        assert m['fallbacks'] == int(frame.fallback.sum())
        assert m['solver_failures'] == int((~frame.success).sum())
        assert_allclose(m['input_bound_active_fraction'], np.any((commands <= lower + 1e-5) | (commands >= upper - 1e-5), axis=1).mean())
        assert_allclose(m['input_slew_active_fraction'], np.any(np.abs(delta) >= cfg['command_delta_per_control'] - 1e-5, axis=1).mean())
        assert_allclose(m['command_variation_squared'], np.sum(delta**2), atol=1e-10)
        error = np.linalg.norm(truth[:, :3] - d['reference'][:, :3], axis=1)
        rmse = np.sqrt(np.mean(error**2))
        complete = n == round(cfg['duration_s'] / dt) and m['termination_reason'] == 'completed'
        assert m['complete'] == complete
        assert_allclose(m['prefix_position_rmse_m'], rmse, atol=1e-12)
        assert_allclose(m['max_position_error_m'], error.max(), atol=1e-12)
        assert_allclose(m['final_position_error_m'], error[-1], atol=1e-12)
        if complete:
            assert_allclose(m['position_rmse_m'], rmse, atol=1e-12)
            assert_allclose(m['late_rmse_m'], np.sqrt(np.mean(error[t >= cfg['duration_s'] / 2]**2)), atol=1e-12)
        else:
            assert m['position_rmse_m'] is None and m['late_rmse_m'] is None
        assert m['tracking_success'] == bool(complete and rmse <= cfg['tracking_success_rmse_m']
            and error.max() <= cfg['tracking_success_max_error_m'] and bounds == slew == 0)
        for field, value in m.items():
            saved = rows.loc[case['case_id'], field] if field != 'case_id' else case['case_id']
            if value is None:
                assert pd.isna(saved)
            elif isinstance(value, (str, bool)):
                assert saved == value
            else:
                assert_allclose(saved, value, atol=1e-12, rtol=1e-12)
        for k in sorted(set([0, n - 1, *range(0, n, 400)])):
            if k < 0:
                continue
            value = solve_ivp(lambda sec, state: derivative(state, u[k], params), [0, dt], truth[k],
                              rtol=2e-11, atol=2e-12).y[:, -1]
            value[6:10] /= np.linalg.norm(value[6:10])
            assert_allclose(value, truth[k + 1], atol=2e-8, rtol=2e-8)
            max_physical = max(max_physical, float(np.max(np.abs(value - truth[k + 1]))))
            counts['independent_plant_steps'] += 1
        counts['cases'] += 1
        counts['completed_cases'] += int(complete)
        counts['observed_transitions'] += n
    result = dict(status='PASS', **counts, max_residual_replay_error=max_residual,
        max_adaptive_state_error=max_physical, preserved_files=check_preserved(),
        only_reference_frequency_changed=True, paired_initial_states_verified=True,
        tracking_sha256=checksum(out / 'tracking_per_run.csv'),
        verification_script_sha256=checksum(Path(__file__)))
    write_json(out / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    with threadpool_limits(limits=1):
        verify(parser.parse_args().run_dir)
