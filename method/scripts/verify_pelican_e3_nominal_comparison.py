"""Independent NumPy replay and paired-data audit for the nominal comparison."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from pelican_e2.data import state_features, advance
from pelican_e2.model import library
from pelican_e3.plant import derivative
from pelican_e3.control import pd_command, bounded_command
from pelican_e3.reference import at
from run_pelican_e2 import checksum, restore_model, write_json
from run_pelican_e3_nominal_comparison import check_preserved


def verify(out):
    out = out.resolve(); frozen = json.loads((out/'FROZEN.json').read_text())
    for rel, value in frozen['source_sha256'].items(): assert checksum(ROOT/rel) == value, rel
    for rel, value in frozen['artifact_sha256'].items(): assert checksum(out/rel) == value, rel
    if frozen.get('development'):
        development = frozen['development']
        assert checksum(ROOT/development['path']/'verification.json') == development['verification_sha256']
    completed = json.loads((out/'COMPLETED.json').read_text())
    assert checksum(out/'tracking_per_run.csv') == completed['tracking_sha256']
    cfg = json.loads((out/'e3_config.json').read_text()); settings = json.loads((out/'config.json').read_text())
    params = json.loads((out/'plant_parameters.json').read_text()); motor = json.loads((out/'motor_model.json').read_text())
    source = ROOT/settings['base_run']
    assert params == json.loads((source/'plant_parameters.json').read_text())
    assert motor == json.loads((source/'motor_model.json').read_text())
    original = {n: restore_model(m) for n, m in json.loads((out/'original_models.json').read_text()).items()}
    models = {n: restore_model(m) for n, m in json.loads((out/'corrected_models.json').read_text()).items()}
    meta = json.loads((out/'equilibrium_calibration.json').read_text())
    original_cfg = json.loads((ROOT/settings['base_run']/'config.json').read_text())
    for key, value in original_cfg.items():
        if key != 'duration_s': assert cfg[key] == value, key
    source_observer = json.loads((ROOT/settings['compensation_source']).read_text())['observer']
    assert settings['observer'] == source_observer
    poles = np.asarray(motor['poles']); gain, offset = np.asarray(motor['coefficients']).T
    uhover = np.asarray(params['hover_command']); steady_motor = gain*uhover+offset
    hover_errors = {}
    for name, model in models.items():
        for key in model:
            if key != 'y_mu': assert_array_equal(model[key], original[name][key])
        assert_allclose(original[name]['y_mu']-model['y_mu'], meta[name]['removed_acceleration_bias'], atol=1e-13)
        x = np.asarray(meta[name]['hover_state']); inp = steady_motor if model['motor'] else uhover
        z = (np.r_[state_features(x, cfg['gravity']), inp]-model['mu'])/model['scale']
        phi = library(z[None], {lag: z[None] for lag in model['lags']}, model['lags'], model['degree'])[0]
        target = (model['coefficients']@(phi/model['feature_scale']))*model['y_scale']+model['y_mu']
        error = target-np.r_[0., 0., cfg['gravity'], 0., 0., 0.]
        assert_allclose(error, 0, atol=1e-11); hover_errors[name] = float(np.max(np.abs(error)))
    plan = json.loads((out/'planned_cases.json').read_text()); summary = pd.read_csv(out/'tracking_per_run.csv')
    assert len(plan) == len(summary) == completed['cases']
    expected = {(r['controller'], r['trajectory'], r['seed']) for r in plan}
    assert len(expected) == len(plan)
    design = settings[frozen['stage']]
    assert expected == {(name, kind, seed) for name in settings['controllers']
                        for kind in design['trajectories'] for seed in design['seeds']}
    assert cfg['duration_s'] == design['duration_s']
    assert not set(settings['development']['seeds']) & set(settings['formal']['seeds'])
    initial_vectors = []
    for seed in design['seeds']:
        with np.load(out/'initial_conditions'/f'{seed}.npz') as f:
            assert int(f['seed']) == seed
            initial_vectors.append(f['truth'][0])
    assert len(np.unique(np.asarray(initial_vectors), axis=0)) == len(design['seeds'])
    assert set(summary[['controller', 'trajectory', 'seed']].itertuples(index=False, name=None)) == expected
    dt = cfg['dt']; hold = round(cfg['control_dt']/dt)
    alpha = 1-np.exp(-dt/np.asarray(settings['observer']['time_constant_s']))
    limit = np.asarray(settings['observer']['estimate_limit']); gate = np.asarray(settings['observer']['residual_gate'])
    counts = dict(cases=0, completed_cases=0, observed_transitions=0, residual_rows=0,
                  causal_bias_checks=0, timing_checks=0, independent_plant_steps=0, pd_command_checks=0)
    max_residual = 0.; max_integration = 0.
    for case in plan:
        name, kind, seed = case['controller'], case['trajectory'], case['seed']
        folder = out/'cases'/f'{kind}_{seed}_{name}'
        with np.load(folder/'trajectory.npz') as f: d = {k: f[k] for k in f.files}
        with np.load(out/'initial_conditions'/f'{seed}.npz') as f: initial = {k: f[k] for k in f.files}
        metrics = json.loads((folder/'metrics.json').read_text()); frame = pd.read_csv(folder/'control.csv')
        x, truth, u, t = d['observation'], d['physical'], d['command'], d['t']
        n = len(u); n0 = len(initial['commands']); mpc = name != 'pd'
        assert len(t) == n+1 and len(x) == n+1
        assert_array_equal(truth[0], initial['truth'][-1]); assert_array_equal(x[0], initial['states'][-1])
        assert_array_equal(d['initial_history_states'], initial['states'])
        assert_array_equal(d['initial_history_commands'], initial['commands'])
        assert_allclose(t, np.arange(n+1)*dt, atol=1e-14)
        assert_allclose(x[1:, :3], truth[1:, :3], atol=1e-14)
        assert_allclose(x[1:, 3:6], np.diff(truth[:, :3], axis=0)/dt, atol=1e-12)
        assert_allclose(x[:, 6:10], truth[:, 6:10], atol=1e-14)
        rates = (Rotation.from_quat(truth[:-1, 6:10]).inv()*Rotation.from_quat(truth[1:, 6:10])).as_rotvec()/dt
        assert_allclose(x[1:, 10:13], rates, atol=1e-12)
        refs = np.array([at(second, kind, cfg)[0] for second in t]); assert_allclose(refs, d['reference'], atol=1e-12)
        estimates = np.zeros((n+1, 6)); accepted = []; clipped = []
        if mpc:
            model = models[name]
            xall = np.vstack([initial['states'], x[1:]]); uall = np.vstack([initial['commands'], u])
            if model['motor']:
                previous = steady_motor.copy(); mhats = []
                for command in uall:
                    previous = poles*previous+(1-poles)*(gain*command+offset); mhats.append(previous.copy())
                mhats = np.asarray(mhats)
                assert_allclose(mhats[n0-1], initial['motor_prior'], atol=1e-12)
                assert_allclose(d['motor_prior'], mhats[n0-1:], atol=1e-12)
                inp = mhats
            else:
                inp = uall
                assert_allclose(d['motor_prior'], np.tile(initial['motor_prior'], (n+1, 1)), atol=1e-12)
            z = (np.c_[state_features(xall[:-1], cfg['gravity']), inp]-model['mu'])/model['scale']
            idx = n0+np.arange(n)
            phi = library(z[idx], {lag: z[idx-lag] for lag in model['lags']}, model['lags'], model['degree'])
            target = ((phi/model['feature_scale'])@model['coefficients'].T)*model['y_scale']+model['y_mu']
            predicted = advance(x[:-1], target, cfg)
            raw = np.c_[Rotation.from_quat(x[:-1, 6:10]).inv().apply((x[1:, 3:6]-predicted[:, 3:6])/dt),
                        (x[1:, 10:13]-predicted[:, 10:13])/dt]
            assert_allclose(raw, d['residual'], atol=1e-10, rtol=1e-10)
            max_residual = max(max_residual, float(np.max(np.abs(raw-d['residual']))))
            for k, residual in enumerate(raw):
                good = bool(np.isfinite(residual).all() and np.all(np.abs(residual) <= gate))
                candidate = estimates[k]+alpha*(residual-estimates[k]) if good else estimates[k]
                estimates[k+1] = np.clip(candidate, -limit, limit)
                accepted.append(good); clipped.append(bool(np.any(candidate != estimates[k+1])))
            counts['residual_rows'] += n
        else:
            assert_allclose(d['residual'], 0, atol=0)
            accepted, clipped = [True]*n, [False]*n
        assert_allclose(estimates, d['bias_estimate'], atol=1e-10)
        assert_array_equal(accepted, d['accepted']); assert_array_equal(clipped, d['clipped'])
        indices = frame.step.to_numpy(dtype=int)
        if metrics['complete']: assert_array_equal(indices, np.arange(0, n, hold))
        assert_allclose(frame[[f'bias{j}' for j in range(6)]], estimates[indices], atol=1e-10)
        commands = frame[[f'u{j}' for j in range(1, 5)]].to_numpy()
        assert_allclose(np.repeat(commands, hold, axis=0)[:n], u, atol=1e-12)
        changes = np.diff(np.vstack([initial['commands'][-1], commands]), axis=0)
        delta = np.abs(changes).max(1); assert_allclose(delta, frame.max_command_change, atol=1e-12)
        assert_allclose(np.sum(changes**2), metrics['command_variation_squared'], atol=1e-10)
        bound_violations = int(np.sum(np.any((u < np.asarray(params['command_lower'])-1e-7)
                                           | (u > np.asarray(params['command_upper'])+1e-7), axis=1)))
        assert metrics['input_bound_violations'] == bound_violations
        assert metrics['input_slew_violations'] == int(np.sum(delta > cfg['command_delta_per_control']+1e-7))
        assert metrics['solver_failures'] == int((~frame.success).sum())
        assert metrics['fallbacks'] == int(frame.fallback.sum())
        assert metrics['observer_rejected'] == int(np.sum(~np.asarray(accepted)))
        assert metrics['observer_clipped'] == int(np.sum(clipped))
        assert np.all(d['pack_seconds'] >= 0) and np.all(d['feedback_seconds'] >= 0)
        for record in frame.itertuples(index=False):
            k = record.step; start = max(0, k-hold+1)
            expected_time = record.command_compute_s+np.sum(d['pack_seconds'][start:k])+np.sum(d['feedback_seconds'][start:k+1])
            assert_allclose(record.full_control_period_s, expected_time, atol=1e-12)
            assert record.feedback_sample_count == min(k, hold)
            if name == 'pd' or record.fallback:
                ref, acc = at(k*dt, kind, cfg)
                old = initial['commands'][-1] if k == 0 else u[k-1]
                expected_u = bounded_command(pd_command(x[k], ref, acc, params, cfg), old, params, cfg)
                assert_allclose(expected_u, commands[k//hold], atol=1e-10)
                counts['pd_command_checks'] += 1
        for field, value in [('full_control_period_median_s', frame.full_control_period_s.median()),
                              ('full_control_period_p95_s', frame.full_control_period_s.quantile(.95)),
                              ('full_control_period_p99_s', frame.full_control_period_s.quantile(.99)),
                              ('full_control_period_max_s', frame.full_control_period_s.max()),
                              ('full_control_over_period_fraction', (frame.full_control_period_s > cfg['control_dt']).mean())]:
            assert_allclose(metrics[field], value, atol=1e-12)
        error = truth[:, :3]-refs[:, :3]; lengths = np.linalg.norm(error, axis=1)
        complete = n == round(cfg['duration_s']/dt) and metrics['termination_reason'] == 'completed'
        assert metrics['complete'] == complete
        rms = float(np.sqrt(np.mean(lengths**2))); late = t >= cfg['duration_s']/2
        assert_allclose(metrics['prefix_position_rmse_m'], rms, atol=1e-12)
        row = summary[(summary.controller == name) & (summary.trajectory == kind) & (summary.seed == seed)].iloc[0]
        assert bool(row.complete) == complete
        # The aggregate table must faithfully copy every per-case metric, including failures.
        for field, value in metrics.items():
            if field not in summary.columns:
                continue
            if value is None:
                assert pd.isna(row[field]), (name, kind, seed, field)
            elif isinstance(value, (str, bool)):
                assert row[field] == value, (name, kind, seed, field)
            elif isinstance(value, (float, int)):
                assert_allclose(row[field], value, atol=1e-12, rtol=1e-12,
                                err_msg=f'{name}/{kind}/{seed}/{field}')
        if complete:
            values = dict(position_rmse_m=rms, late_rmse_m=np.sqrt(np.mean(lengths[late]**2)),
                late_bias_norm_m=np.linalg.norm(error[late].mean(0)), max_position_error_m=lengths.max(), final_position_error_m=lengths[-1])
            for field, value in values.items():
                assert_allclose(metrics[field], value, atol=1e-12); assert_allclose(row[field], value, atol=1e-12)
        else:
            assert metrics['position_rmse_m'] is None and pd.isna(row.position_rmse_m)
        success = complete and rms <= cfg['tracking_success_rmse_m'] and lengths.max() <= cfg['tracking_success_max_error_m'] and bound_violations == metrics['input_slew_violations'] == 0
        assert metrics['tracking_success'] == success
        for k in sorted(set([0, n-1, *range(0, n, 400)])):
            if k < 0: continue
            value = solve_ivp(lambda sec, state: derivative(state, u[k], params), [0, dt], truth[k],
                              rtol=2e-11, atol=2e-12).y[:, -1]
            value[6:10] /= np.linalg.norm(value[6:10])
            assert_allclose(value, truth[k+1], atol=2e-8, rtol=2e-8)
            max_integration = max(max_integration, float(np.max(np.abs(value-truth[k+1]))))
            counts['independent_plant_steps'] += 1
        counts['cases'] += 1; counts['completed_cases'] += int(complete)
        counts['observed_transitions'] += n; counts['timing_checks'] += len(frame)
        counts['causal_bias_checks'] += len(frame) if mpc else 0
    result = dict(status='PASS', **counts, max_residual_replay_error=max_residual,
                  max_adaptive_state_error=max_integration, hover_output_errors=hover_errors,
                  preserved_files=check_preserved(), tracking_sha256=checksum(out/'tracking_per_run.csv'),
                  verification_script_sha256=checksum(Path(__file__)))
    write_json(out/'verification.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', type=Path, required=True)
    with threadpool_limits(limits=1): verify(parser.parse_args().run_dir)
