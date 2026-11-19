"""Independent batch replay of residuals, causal filtering and experiment metrics."""
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
from pelican_e3.reference import at
from run_pelican_e2 import checksum, restore_model, write_json
from run_pelican_e3_corrected import check_originals


def verify(out):
    frozen = json.loads((out/'FROZEN.json').read_text())
    for rel, value in frozen['source_sha256'].items():
        assert checksum(ROOT/rel) == value, rel
    for rel, value in frozen['artifact_sha256'].items():
        assert checksum(out/rel) == value, rel
    completed = json.loads((out/'COMPLETED.json').read_text())
    assert checksum(out/'tracking_per_run.csv') == completed['tracking_sha256']
    cfg = json.loads((out/'e3_config.json').read_text())
    settings = json.loads((out/'config.json').read_text())
    params = json.loads((out/'plant_parameters.json').read_text())
    original = restore_model(json.loads((out/'original_model.json').read_text()))
    corrected = restore_model(json.loads((out/'corrected_model.json').read_text()))
    meta = json.loads((out/'equilibrium_calibration.json').read_text())
    base_cfg = json.loads((ROOT/settings['base_run']/'config.json').read_text())
    for key, value in base_cfg.items():
        if key != 'duration_s':
            assert cfg[key] == value, 'MPC/plant setting changed: '+key
    for key in original:
        if key != 'y_mu':
            assert_array_equal(original[key], corrected[key])
    assert_allclose(original['y_mu']-corrected['y_mu'], meta['removed_acceleration_bias'], atol=1e-14)
    hover_x = np.asarray(meta['hover_state']); hover_u = np.asarray(meta['hover_command'])
    zhover = (np.r_[state_features(hover_x, cfg['gravity']), hover_u]-corrected['mu'])/corrected['scale']
    phover = library(zhover[None], {lag: zhover[None] for lag in corrected['lags']}, corrected['lags'], corrected['degree'])[0]
    hover_output = (corrected['coefficients']@(phover/corrected['feature_scale']))*corrected['y_scale']+corrected['y_mu']
    hover_error = hover_output-np.array([0, 0, cfg['gravity'], 0, 0, 0])
    assert_allclose(hover_error, 0, atol=1e-11)
    physical_hover = np.r_[hover_x, np.asarray(params['motor_gain'])*hover_u+params['motor_bias']]
    assert_allclose(derivative(physical_hover, hover_u, params), 0, atol=1e-11)
    plan = json.loads((out/'planned_cases.json').read_text())
    summary = pd.read_csv(out/'tracking_per_run.csv')
    assert len(summary) == len(plan) == completed['cases']
    expected_keys = {(r['scenario'], r['trajectory'], r['seed'], r['variant']) for r in plan}
    assert len(expected_keys) == len(plan)
    assert set(summary[['scenario', 'trajectory', 'seed', 'variant']].itertuples(index=False, name=None)) == expected_keys
    dt = cfg['dt']; hold = round(cfg['control_dt']/dt)
    alpha = 1-np.exp(-dt/np.asarray(settings['observer']['time_constant_s']))
    limit = np.asarray(settings['observer']['estimate_limit']); gate = np.asarray(settings['observer']['residual_gate'])
    counter = dict(cases=0, residual_rows=0, control_bias_checks=0, adaptive_plant_steps=0)
    max_residual_error = 0.; max_physical_error = 0.
    for case in plan:
        name = '{scenario}_{trajectory}_{seed}_{variant}'.format(**case)
        folder = out/'cases'/name
        metrics = json.loads((folder/'metrics.json').read_text())
        with np.load(folder/'trajectory.npz') as f: data = {key: f[key] for key in f.files}
        with np.load(out/'initial_conditions'/f"{case['seed']}.npz") as f: initial = {key: f[key] for key in f.files}
        frame = pd.read_csv(folder/'control.csv')
        x, truth, u, t = data['observation'], data['physical'], data['command'], data['t']
        nsteps = len(u); n0 = len(initial['commands'])
        assert metrics['complete'] and nsteps == round(cfg['duration_s']/dt)
        assert_array_equal(data['initial_history_states'], initial['states'])
        assert_array_equal(data['initial_history_commands'], initial['commands'])
        assert_array_equal(truth[0], initial['truth'][-1]); assert_array_equal(x[0], initial['states'][-1])
        assert_allclose(t, np.arange(nsteps+1)*dt, atol=1e-14)
        assert_allclose(x[1:, :3], truth[1:, :3], atol=1e-14)
        assert_allclose(x[1:, 3:6], np.diff(truth[:, :3], axis=0)/dt, atol=1e-12)
        rates = (Rotation.from_quat(truth[:-1, 6:10]).inv()*Rotation.from_quat(truth[1:, 6:10])).as_rotvec()/dt
        assert_allclose(x[1:, 10:13], rates, atol=1e-12)
        assert_allclose(x[:, 6:10], truth[:, 6:10], atol=1e-14)
        refs = np.array([at(second, case['trajectory'], cfg)[0] for second in t])
        assert_allclose(data['reference'], refs, atol=1e-12)
        model = corrected if case['variant'] in ['equilibrium', 'combined'] else original
        assert not model['motor']
        xall = np.vstack([initial['states'], x[1:]])
        uall = np.vstack([initial['commands'], u])
        z = (np.c_[state_features(xall[:-1], cfg['gravity']), uall]-model['mu'])/model['scale']
        idx = n0+np.arange(nsteps)
        phi = library(z[idx], {lag: z[idx-lag] for lag in model['lags']}, model['lags'], model['degree'])
        target = ((phi/model['feature_scale'])@model['coefficients'].T)*model['y_scale']+model['y_mu']
        predicted = advance(x[:-1], target, cfg)
        raw = np.c_[Rotation.from_quat(x[:-1, 6:10]).inv().apply((x[1:, 3:6]-predicted[:, 3:6])/dt),
                    (x[1:, 10:13]-predicted[:, 10:13])/dt]
        assert_allclose(raw, data['residual'], atol=1e-10, rtol=1e-10)
        max_residual_error = max(max_residual_error, float(np.max(np.abs(raw-data['residual']))))
        enabled = case['variant'] in ['online', 'combined']
        estimates = np.zeros((nsteps+1, 6)); accepted = []; clipped = []
        for k, residual in enumerate(raw):
            good = bool(np.isfinite(residual).all() and np.all(np.abs(residual) <= gate))
            candidate = estimates[k]+alpha*(residual-estimates[k]) if good else estimates[k]
            projected = np.clip(candidate, -limit, limit)
            estimates[k+1] = projected if enabled else 0
            accepted.append(good); clipped.append(bool(np.any(candidate != projected)) if enabled else False)
        assert_allclose(estimates, data['bias_estimate'], atol=1e-10, rtol=1e-10)
        assert_array_equal(accepted, data['accepted']); assert_array_equal(clipped, data['clipped'])
        indices = frame.step.to_numpy(dtype=int)
        assert_array_equal(indices, np.arange(0, nsteps, hold))
        # Estimate indexed by k contains only observations through k, not k+1.
        assert_allclose(frame[[f'bias{i}' for i in range(6)]], estimates[indices], atol=1e-10)
        command_rows = frame[[f'u{i}' for i in range(1, 5)]].to_numpy()
        assert_allclose(u, np.repeat(command_rows, hold, axis=0), atol=1e-12)
        delta = np.abs(np.diff(np.vstack([initial['commands'][-1], command_rows]), axis=0)).max(1)
        assert_allclose(delta, frame.max_command_change, atol=1e-12)
        violations = int(np.sum(np.any((u < np.asarray(params['command_lower'])-1e-7)
                                      | (u > np.asarray(params['command_upper'])+1e-7), axis=1)))
        assert violations == metrics['input_bound_violations']
        assert int(np.sum(delta > cfg['command_delta_per_control']+1e-7)) == metrics['input_slew_violations']
        assert int((~frame.success).sum()) == metrics['solver_failures']
        assert int(frame.fallback.sum()) == metrics['fallbacks']
        assert int(np.sum(~np.array(accepted))) == metrics['observer_rejected']
        assert int(np.sum(clipped)) == metrics['observer_clipped']
        error = truth[:, :3]-refs[:, :3]; norm = np.linalg.norm(error, axis=1); late = t >= cfg['duration_s']/2
        calculated = dict(position_rmse_m=np.sqrt(np.mean(norm**2)), late_rmse_m=np.sqrt(np.mean(norm[late]**2)),
                          late_bias_norm_m=np.linalg.norm(error[late].mean(0)), max_position_error_m=norm.max(),
                          final_position_error_m=norm[-1])
        row = summary[(summary.scenario == case['scenario']) & (summary.trajectory == case['trajectory'])
                      & (summary.seed == case['seed']) & (summary.variant == case['variant'])].iloc[0]
        for key, value in calculated.items():
            assert_allclose(metrics[key], value, atol=1e-12); assert_allclose(row[key], value, atol=1e-12)
        extra = np.zeros((nsteps, 6))
        if case['scenario'] == 'bias_step':
            extra[t[:-1] >= settings['disturbance']['start_s']] = settings['disturbance']['body_acceleration']
        assert_array_equal(extra, data['injected_body_acceleration'])
        # External adaptive integrator, separate from new RK4 implementation.
        selected = sorted(set([0, nsteps-1, *range(0, nsteps, 250),
                               round(settings['disturbance']['start_s']/dt)]))
        for k in selected:
            if k >= nsteps: continue
            d = extra[k]
            def rhs(second, state):
                value = derivative(state, u[k], params)
                value[3:6] += Rotation.from_quat(state[6:10]).apply(d[:3]); value[10:13] += d[3:]
                return value
            sol = solve_ivp(rhs, [0, dt], truth[k], rtol=2e-11, atol=2e-12).y[:, -1]
            sol[6:10] /= np.linalg.norm(sol[6:10])
            assert_allclose(sol, truth[k+1], atol=2e-8, rtol=2e-8)
            max_physical_error = max(max_physical_error, float(np.max(np.abs(sol-truth[k+1]))))
            counter['adaptive_plant_steps'] += 1
        counter['cases'] += 1; counter['residual_rows'] += nsteps; counter['control_bias_checks'] += len(frame)
    result = dict(status='PASS', **counter, max_residual_replay_error=max_residual_error,
                  calibrated_hover_output_max_error=float(np.max(np.abs(hover_error))),
                  max_adaptive_state_error=max_physical_error, original_artifacts_verified=check_originals(),
                  tracking_sha256=checksum(out/'tracking_per_run.csv'),
                  verification_script_sha256=checksum(Path(__file__)),
                  scope='Independent NumPy batch replay, causal estimate indexing, metrics, bounds, paired initial states and adaptive plant checks')
    write_json(out/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', type=Path, required=True)
    with threadpool_limits(limits=1):
        verify(parser.parse_args().run_dir)
