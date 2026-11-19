"""Independent observation, residual and integration replay for E4 estimation diagnosis."""
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
from run_pelican_e4_estimation_diagnostic import preserved_e4


def independent_filter(pos, quat, dt, window):
    result = np.zeros((len(pos),13))
    for k in range(len(pos)):
        if k == 0:
            result[k,:3] = pos[k]; result[k,6:10] = quat[k]; continue
        start = max(0,k-window+1)
        times = np.arange(start-k,1,dtype=float)
        degree = min(2,len(times)-1)
        design = np.column_stack([times**j for j in range(degree+1)])
        pc = np.linalg.lstsq(design,pos[start:k+1],rcond=None)[0]
        anchor = Rotation.from_quat(quat[k])
        coords = (anchor.inv()*Rotation.from_quat(quat[start:k+1])).as_rotvec()
        rc = np.linalg.lstsq(design,coords,rcond=None)[0]
        old = (-1.)**np.arange(degree+1)
        rnow = anchor*Rotation.from_rotvec(rc[0])
        rprev = anchor*Rotation.from_rotvec(old@rc)
        result[k] = np.r_[pc[0],(pc[0]-old@pc)/dt,rnow.as_quat(),(rprev.inv()*rnow).as_rotvec()/dt]
    return result


def verify(out):
    out=out.resolve(); frozen=json.loads((out/'FROZEN.json').read_text())
    for rel,h in frozen['source_sha256'].items(): assert checksum(ROOT/rel)==h,rel
    for rel,h in frozen['artifact_sha256'].items(): assert checksum(out/rel)==h,rel
    design=json.loads((out/'design.json').read_text())
    source=ROOT/design['base_run']
    assert frozen['base_manifest_sha256']==checksum(source/'artifact_hashes.json')
    preserved=preserved_e4(source)
    for file in ['plant_parameters.json','motor_model.json','corrected_models.json','equilibrium_calibration.json','e3_config.json']:
        assert checksum(out/file)==checksum(source/file),file
    cfg=json.loads((out/'e3_config.json').read_text()); settings=json.loads((out/'config.json').read_text())
    original_settings=json.loads((source/'config.json').read_text())
    assert all(settings[k]==v for k,v in original_settings.items() if k!='stage_runtime_limit_s')
    params=json.loads((out/'plant_parameters.json').read_text());motor=json.loads((out/'motor_model.json').read_text())
    models={k:restore_model(v) for k,v in json.loads((out/'corrected_models.json').read_text()).items()}
    plan=json.loads((out/'planned_cases.json').read_text());summary=pd.read_csv(out/'diagnostic_per_run.csv')
    completed=json.loads((out/'COMPLETED.json').read_text())
    sanity=design.get('no_noise_filter_check',False)
    assert len(plan)==len(summary)==completed['cases']==(8 if sanity else 48)
    assert completed['tracking_sha256']==checksum(out/'diagnostic_per_run.csv')
    assert set(summary.case_id)=={c['case_id'] for c in plan}
    expected={(m,True,n,t,s) for m in design['primary_modes'] for n in design['controllers'] for t in design['trajectories'] for s in design['seeds']}
    expected|={(m,False,n,t,design['observer_ablation_seed']) for m in design['observer_ablation_modes'] for n in design['observer_ablation_controllers'] for t in design['trajectories']}
    assert {(c['mode'],c['bias_enabled'],c['controller'],c['trajectory'],c['seed']) for c in plan}==expected
    dt=cfg['dt'];hold=round(cfg['control_dt']/dt)
    scenario=next(s for s in settings['scenarios'] if s['name']==('nominal' if sanity else 'pose_noise'))
    poles=np.asarray(motor['poles']);gain,offset=np.asarray(motor['coefficients']).T
    steady_motor=gain*np.asarray(params['hover_command'])+offset
    alpha=1-np.exp(-dt/np.asarray(settings['observer']['time_constant_s']))
    limit=np.asarray(settings['observer']['estimate_limit']);gate=np.asarray(settings['observer']['residual_gate'])
    counts=dict(cases=0,completed_cases=0,observed_transitions=0,residual_rows=0,causal_bias_checks=0,timing_checks=0,independent_plant_steps=0,pd_command_checks=0)
    max_residual=0.;max_integration=0.;max_observation=0.
    for case in plan:
        name,kind,seed=(case[k] for k in ['controller','trajectory','seed'])
        scenario_name=scenario['name'];actual_params=params
        folder=out/'cases'/case['case_id']
        with np.load(folder/'trajectory.npz') as f: d = {k: f[k] for k in f.files}
        with np.load(out/'initial_conditions'/f'{seed}.npz') as f: initial = {k: f[k] for k in f.files}
        metrics = json.loads((folder/'metrics.json').read_text()); frame = pd.read_csv(folder/'control.csv')
        x, truth, u, t = d['observation'], d['physical'], d['command'], d['t']
        n = len(u); n0 = len(initial['commands']); mpc = name != 'pd'
        assert len(t) == n+1 and len(x) == n+1
        assert_array_equal(truth[0], initial['truth'][-1])
        assert_array_equal(d['initial_history_commands'], initial['commands'])
        assert_allclose(t, np.arange(n+1)*dt, atol=1e-14)
        count = len(initial['truth'])+round(cfg['duration_s']/dt)
        standardized = np.random.default_rng(settings['noise_seed_offset']+seed).standard_normal((count,6))
        noise = standardized*np.r_[np.full(3,scenario['position_noise_std_m']),np.full(3,scenario['rotation_noise_std_rad'])]
        assert_array_equal(d['pose_noise'],noise)
        whole_truth = np.vstack([initial['truth'],truth[1:]])
        pos = whole_truth[:,:3]+noise[:len(whole_truth),:3]
        quat = (Rotation.from_quat(whole_truth[:,6:10])*Rotation.from_rotvec(noise[:len(whole_truth),3:])).as_quat()
        all_observations = whole_truth[:,:13].copy()
        all_observations[:,:3] = pos; all_observations[:,6:10] = quat
        all_observations[1:,3:6] = np.diff(pos,axis=0)/dt
        all_observations[1:,10:13] = (Rotation.from_quat(quat[:-1]).inv()*Rotation.from_quat(quat[1:])).as_rotvec()/dt
        clean=whole_truth[:,:13].copy()
        clean[1:,3:6]=np.diff(whole_truth[:,:3],axis=0)/dt
        clean[1:,10:13]=(Rotation.from_quat(whole_truth[:-1,6:10]).inv()*Rotation.from_quat(whole_truth[1:,6:10])).as_rotvec()/dt
        assert_allclose(d['raw_observation'],all_observations[n0:],atol=1e-12)
        assert_allclose(d['clean_observation'],clean[n0:],atol=1e-12)
        if case['mode']=='oracle_rates':
            all_observations[:,3:6]=clean[:,3:6];all_observations[:,10:13]=clean[:,10:13]
        elif case['mode']=='causal_poly':
            all_observations=independent_filter(pos,quat,dt,design['filter_window_samples'])
        observed_initial=all_observations[:n0+1]
        assert_allclose(d['initial_history_states'],observed_initial,atol=2e-11)
        assert_allclose(x,all_observations[n0:],atol=2e-11)
        max_observation=max(max_observation,float(np.max(np.abs(x-all_observations[n0:]))))
        for suffix,mask in [('prefix',np.ones(len(t),bool)),('first_0p4s',t<=.4+1e-12)]:
            expected_values={
                'velocity':np.sqrt(np.mean(np.sum((x[mask,3:6]-clean[n0:][mask,3:6])**2,axis=1))),
                'rate':np.sqrt(np.mean(np.sum((x[mask,10:13]-clean[n0:][mask,10:13])**2,axis=1))),
                'position':np.sqrt(np.mean(np.sum((x[mask,:3]-truth[mask,:3])**2,axis=1))),
                'attitude':np.sqrt(np.mean((Rotation.from_quat(truth[mask,6:10]).inv()*Rotation.from_quat(x[mask,6:10])).magnitude()**2))}
            for label,value in expected_values.items():assert_allclose(metrics[label+'_observation_rmse_'+suffix],value,atol=2e-11)
        assert metrics['mode']==case['mode'] and metrics['bias_enabled']==case['bias_enabled']
        acceleration = np.zeros((n,3))
        active = (t[:-1] >= settings['force_start_s']) & (t[:-1] < settings['force_end_s'])
        acceleration[active] = scenario['world_acceleration_m_s2']
        assert_array_equal(d['world_acceleration'],acceleration)
        refs = np.array([at(second, kind, cfg)[0] for second in t]); assert_allclose(refs, d['reference'], atol=1e-12)
        estimates = np.zeros((n+1, 6)); accepted = []; clipped = []
        if mpc:
            model = models[name]
            xall = np.vstack([d['initial_history_states'], x[1:]]); uall = np.vstack([initial['commands'], u])
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
        assert_allclose(frame[[f'bias{j}' for j in range(6)]], estimates[indices] if case['bias_enabled'] else 0., atol=1e-10)
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
        tilt=np.degrees(np.arccos(np.clip(Rotation.from_quat(truth[1:,6:10]).as_matrix()[:,2,2],-1,1)))
        reason='completed'
        for j in range(1,len(truth)):
            if truth[j,2]<cfg['termination_min_height_m']:reason='minimum_height'
            elif lengths[j]>cfg['termination_position_error_m']:reason='position_error'
            elif tilt[j-1]>cfg['termination_tilt_deg']:reason='tilt'
            elif np.linalg.norm(truth[j,3:6])>cfg['termination_speed_m_s']:reason='speed'
            elif np.linalg.norm(truth[j,10:13])>cfg['termination_rate_rad_s']:reason='angular_rate'
            if reason!='completed':assert j==len(truth)-1;break
        assert reason==metrics['termination_reason']

        rms = float(np.sqrt(np.mean(lengths**2))); late = t >= cfg['duration_s']/2
        assert_allclose(metrics['prefix_position_rmse_m'], rms, atol=1e-12)
        row = summary[summary.case_id==case['case_id']].iloc[0]
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
        event = (t >= settings['force_start_s']) & (t <= settings['force_end_s'])
        peak = float(lengths[event].max()) if event.any() else None
        if peak is None: assert metrics['event_peak_error_m'] is None
        else: assert_allclose(metrics['event_peak_error_m'],peak,atol=1e-12)
        after = t >= settings['post_event_start_s']
        if complete and after.any(): assert_allclose(metrics['post_event_rmse_m'],np.sqrt(np.mean(lengths[after]**2)),atol=1e-12)
        else: assert metrics['post_event_rmse_m'] is None
        window = round(settings['recovery_dwell_s']/dt)+1
        good = np.convolve((lengths <= settings['recovery_error_m']).astype(int),np.ones(window,dtype=int),'valid') if len(lengths)>=window else np.array([])
        starts = np.flatnonzero((good==window) & (t[:len(good)]>=settings['force_end_s']))
        assert metrics['recovered'] == bool(len(starts))
        if len(starts): assert_allclose(metrics['recovery_time_s'],t[starts[0]]-settings['force_end_s'],atol=1e-12)
        else: assert metrics['recovery_time_s'] is None
        sample_indices = {0,n-1,*range(0,n,400),799,800,1199,1200}
        for k in sorted(i for i in sample_indices if 0 <= i < n):
            if k < 0: continue
            def rhs(sec,state):
                value=derivative(state,u[k],actual_params); value[3:6]+=acceleration[k]; return value
            value = solve_ivp(rhs, [0, dt], truth[k], rtol=2e-11, atol=2e-12).y[:, -1]
            value[6:10] /= np.linalg.norm(value[6:10])
            assert_allclose(value, truth[k+1], atol=2e-8, rtol=2e-8)
            max_integration = max(max_integration, float(np.max(np.abs(value-truth[k+1]))))
            counts['independent_plant_steps'] += 1
        counts['cases'] += 1; counts['completed_cases'] += int(complete)
        counts['observed_transitions'] += n; counts['timing_checks'] += len(frame)
        counts['causal_bias_checks'] += len(frame) if mpc else 0
    result=dict(status='PASS',**counts,max_observation_replay_error=max_observation,max_residual_replay_error=max_residual,
                max_adaptive_state_error=max_integration,preserved_files=preserved,
                tracking_sha256=checksum(out/'diagnostic_per_run.csv'),verification_script_sha256=checksum(Path(__file__)))
    write_json(out/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    with threadpool_limits(limits=1):verify(parser.parse_args().run_dir)
