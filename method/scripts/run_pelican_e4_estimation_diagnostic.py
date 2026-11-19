"""Exploratory E4 observation diagnosis, derived from the frozen E4 loop.

The control loop retains E3 ordering and timing; zero-intervention equivalence
is checked in tests. Old files and all frozen model artifacts are untouched.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
import importlib.metadata
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from pelican_e3.control import pack_observation, pd_command, bounded_command
from pelican_e3.plant import observe
from pelican_e3.reference import at
from pelican_e3_corrected.control import BiasMPC, BiasObserver
from pelican_e4.environment import plant_parameters, physical_step, forcing, noise_samples, measured_pose, observation_history, event_metrics
from run_pelican_e2 import restore_model, checksum, write_json
from run_pelican_e3 import initialize
from run_pelican_e3_nominal_comparison import full_period_time, log, check_preserved as old_preserved


from pelican_e4_estimation.observation import ObservationPipeline


def run_case(scenario, name, kind, seed, controller, initial, parameters, motor, cfg,
             settings, case_dir, logfn, stage_started, mode="raw", bias_enabled=True, window=11):
    noise = noise_samples(seed, len(initial['truth'])+round(cfg['duration_s']/cfg['dt']), scenario, settings)
    pipeline = ObservationPipeline(mode, initial, noise[:len(initial['truth'])], cfg['dt'], window)
    observed_initial = pipeline.initial_states
    clean_log = [initial['states'][-1].copy()]
    raw_log = [pipeline.initial_raw[-1].copy()]
    states = list(observed_initial.copy()); commands = list(initial['commands'].copy())
    actual_parameters = plant_parameters(parameters, scenario)
    physical = initial['truth'][-1].copy(); prior = initial['motor_prior'].copy()
    previous = commands[-1].copy()
    is_mpc = controller is not None
    if is_mpc:
        controller.reset()
        observer = BiasObserver(controller.transition, cfg['dt'], settings['observer'])
    else: observer = None
    hold = round(cfg['control_dt']/cfg['dt']); nsteps = round(cfg['duration_s']/cfg['dt'])
    poles = np.asarray(motor['poles']); gain, offset = np.asarray(motor['coefficients']).T
    references = [at(k*cfg['dt'], kind, cfg) for k in range(nsteps+cfg['prediction_steps']+1)]
    refs = np.array([r[0] for r in references]); accs = np.array([r[1] for r in references])
    acceleration = np.array([forcing(k*cfg['dt'], scenario, settings) for k in range(nsteps)])
    truth = [physical.copy()]; measured = [states[-1].copy()]; inputs = []; controls = []
    estimates = [np.zeros(6)]; residuals = []; accepted_log = []; clipped_log = []
    priors = [prior.copy()]; pack_times = []; feedback_times = [0.]
    reason = 'completed'; started = time.monotonic()
    for k in range(nsteps):
        if time.monotonic()-stage_started > settings['stage_runtime_limit_s']:
            raise TimeoutError('E4 stage runtime limit exceeded')
        command_start = time.perf_counter()
        pack_start = time.perf_counter()
        packed = pack_observation(states, commands, prior, controller.model, cfg) if is_mpc else None
        pack_times.append(time.perf_counter()-pack_start)
        if k % hold == 0:
            used = observer.estimate.copy() if is_mpc and bias_enabled else np.zeros(6)
            if is_mpc:
                horizon = refs[k+np.arange(1, controller.blocks+1)*hold]
                requested, stats = controller.solve(packed, horizon, previous, used)
                fallback = not stats['success']
                if fallback: requested = pd_command(states[-1], refs[k], accs[k], parameters, cfg)
            else:
                requested = pd_command(states[-1], refs[k], accs[k], parameters, cfg)
                stats = dict(success=True, status='pd', seconds=0., iterations=0, objective=None)
                fallback = False
            u = bounded_command(requested, previous, parameters, cfg)
            delta = float(np.max(np.abs(u-previous)))
            adjustment = float(np.max(np.abs(u-requested)))
            previous = u.copy()
            command_seconds = time.perf_counter()-command_start
            controls.append(dict(step=k, t=k*cfg['dt'], **stats, fallback=fallback,
                command_compute_s=command_seconds,
                full_control_period_s=full_period_time(k, hold, command_seconds, pack_times, feedback_times),
                feedback_sample_count=min(k, hold), bounded_adjustment=adjustment, max_command_change=delta,
                **{f'u{j+1}': float(u[j]) for j in range(4)},
                **{f'bias{j}': float(used[j]) for j in range(6)}))
            if k % 1000 == 0:
                logfn('case_progress', controller=name, trajectory=kind, seed=seed,
                      simulated_s=k*cfg['dt'], elapsed_case_s=time.monotonic()-started)
        new = physical_step(physical, u, cfg['dt'], actual_parameters, cfg['integration_dt'], acceleration[k])
        if not np.isfinite(new).all(): reason = 'nonfinite_plant'; break
        feedback_start = time.perf_counter()
        observation, raw_observation, clean_observation = pipeline.update(new, noise[len(initial['truth'])+k])
        raw_log.append(raw_observation); clean_log.append(clean_observation)
        if is_mpc:
            raw, estimate, accepted, clipped = observer.update(packed, u, observation)
        else:
            raw, estimate, accepted, clipped = np.zeros(6), np.zeros(6), True, False
        states.append(observation); commands.append(u.copy())
        if is_mpc and controller.model['motor']:
            prior = poles*prior+(1-poles)*(gain*u+offset)
        feedback_times.append(time.perf_counter()-feedback_start)
        residuals.append(raw); estimates.append(estimate); priors.append(prior.copy())
        accepted_log.append(accepted); clipped_log.append(clipped)
        physical = new
        truth.append(physical.copy()); measured.append(observation.copy()); inputs.append(u.copy())
        error = np.linalg.norm(physical[:3]-refs[k+1, :3])
        tilt = np.degrees(np.arccos(np.clip(Rotation.from_quat(physical[6:10]).as_matrix()[2, 2], -1, 1)))
        if physical[2] < cfg['termination_min_height_m']: reason = 'minimum_height'
        elif error > cfg['termination_position_error_m']: reason = 'position_error'
        elif tilt > cfg['termination_tilt_deg']: reason = 'tilt'
        elif np.linalg.norm(physical[3:6]) > cfg['termination_speed_m_s']: reason = 'speed'
        elif np.linalg.norm(physical[10:13]) > cfg['termination_rate_rad_s']: reason = 'angular_rate'
        if reason != 'completed': break
    truth = np.asarray(truth); reference = refs[:len(truth)]
    t = np.arange(len(truth))*cfg['dt']; errors = np.linalg.norm(truth[:, :3]-reference[:, :3], axis=1)
    frame = pd.DataFrame(controls); applied = np.asarray(inputs).reshape(-1, 4)
    complete = len(inputs) == nsteps and reason == 'completed'
    bounds = int(np.sum(np.any((applied < np.asarray(parameters['command_lower'])-1e-7)
                               | (applied > np.asarray(parameters['command_upper'])+1e-7), axis=1)))
    slew = int(np.sum(frame.max_command_change > cfg['command_delta_per_control']+1e-7))
    rmse = float(np.sqrt(np.mean(errors**2))); late = t >= cfg['duration_s']/2
    changes = np.diff(np.vstack([initial['commands'][-1], frame[[f'u{i}' for i in range(1, 5)]].to_numpy()]), axis=0)
    metrics = dict(scenario=scenario['name'], controller=name, trajectory=kind, seed=seed, complete=complete, termination_reason=reason,
        duration_completed_s=float(t[-1]), requested_duration_s=cfg['duration_s'],
        position_rmse_m=rmse if complete else None, prefix_position_rmse_m=rmse,
        late_rmse_m=float(np.sqrt(np.mean(errors[late]**2))) if complete else None,
        late_bias_norm_m=float(np.linalg.norm((truth[late, :3]-reference[late, :3]).mean(0))) if complete else None,
        max_position_error_m=float(errors.max()), final_position_error_m=float(errors[-1]),
        tracking_success=bool(complete and rmse <= cfg['tracking_success_rmse_m'] and errors.max() <= cfg['tracking_success_max_error_m'] and bounds == slew == 0),
        solver_failures=int((~frame.success).sum()), fallbacks=int(frame.fallback.sum()),
        control_calls=len(frame), input_bound_violations=bounds, input_slew_violations=slew,
        observer_rejected=int(np.sum(~np.asarray(accepted_log, dtype=bool))) if is_mpc else 0,
        observer_clipped=int(np.sum(clipped_log)), command_variation_squared=float(np.sum(changes**2)),
        solve_time_median_s=float(frame.seconds.median()) if is_mpc else None,
        solve_time_p95_s=float(frame.seconds.quantile(.95)) if is_mpc else None,
        full_control_period_median_s=float(frame.full_control_period_s.median()),
        full_control_period_p95_s=float(frame.full_control_period_s.quantile(.95)),
        full_control_period_p99_s=float(frame.full_control_period_s.quantile(.99)),
        full_control_period_max_s=float(frame.full_control_period_s.max()),
        full_control_over_period_fraction=float((frame.full_control_period_s > cfg['control_dt']).mean()),
        case_wall_s=time.monotonic()-started)
    metrics.update(event_metrics(t, errors, complete, settings))
    metrics.update(mode=mode, bias_enabled=bias_enabled, window_samples=window if mode == 'causal_poly' else 0)
    obs, clean = np.asarray(measured), np.asarray(clean_log)
    for suffix, mask in [('prefix', np.ones(len(t), dtype=bool)), ('first_0p4s', t <= .4+1e-12)]:
        metrics['velocity_observation_rmse_'+suffix] = float(np.sqrt(np.mean(np.sum((obs[mask,3:6]-clean[mask,3:6])**2,axis=1))))
        metrics['rate_observation_rmse_'+suffix] = float(np.sqrt(np.mean(np.sum((obs[mask,10:13]-clean[mask,10:13])**2,axis=1))))
        metrics['position_observation_rmse_'+suffix] = float(np.sqrt(np.mean(np.sum((obs[mask,:3]-truth[mask,:3])**2,axis=1))))
        angle = (Rotation.from_quat(truth[mask,6:10]).inv()*Rotation.from_quat(obs[mask,6:10])).magnitude()
        metrics['attitude_observation_rmse_'+suffix] = float(np.sqrt(np.mean(angle**2)))
    case_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(case_dir/'trajectory.npz', t=t, physical=truth, observation=measured,
        reference=reference, command=applied, bias_estimate=estimates, residual=residuals,
        accepted=accepted_log, clipped=clipped_log, motor_prior=priors,
        pack_seconds=pack_times, feedback_seconds=feedback_times,
        initial_history_states=observed_initial, initial_history_commands=initial['commands'],
        raw_observation=raw_log, clean_observation=clean_log,
        pose_noise=noise, world_acceleration=acceleration[:len(inputs)])
    frame.to_csv(case_dir/'control.csv', index=False); write_json(case_dir/'metrics.json', metrics)
    return metrics


def preserved_e4(source):
    counts = old_preserved()
    for rel in ['runs/pelican_e3_nominal_comparison_v1/artifact_hashes.json',
                (source/'artifact_hashes.json').relative_to(ROOT).as_posix()]:
        files = json.loads((ROOT/rel).read_text(encoding='utf-8'))['files']
        for path, item in files.items():
            assert checksum(ROOT/path) == item['sha256'], path
        counts[rel] = len(files)
    return counts


def main(args):
    design = json.loads(args.config.read_text(encoding='utf-8'))
    source = ROOT/design['base_run']
    preserved = preserved_e4(source)
    settings = json.loads((source/'config.json').read_text())
    settings['stage_runtime_limit_s'] = design['stage_runtime_limit_s']
    cfg = json.loads((source/'e3_config.json').read_text())
    assert cfg['duration_s'] == design['duration_s']
    parameters = json.loads((source/'plant_parameters.json').read_text())
    motor = json.loads((source/'motor_model.json').read_text())
    models = {k: restore_model(v) for k,v in json.loads((source/'corrected_models.json').read_text()).items()}
    scenario = next(s for s in settings['scenarios'] if s['name'] == 'pose_noise')
    plan = [dict(mode=mode, bias_enabled=True, controller=name, trajectory=kind, seed=seed)
            for mode in design['primary_modes'] for seed in design['seeds']
            for kind in design['trajectories'] for name in design['controllers']]
    plan += [dict(mode=mode, bias_enabled=False, controller=name, trajectory=kind,
                  seed=design['observer_ablation_seed'])
             for mode in design['observer_ablation_modes'] for kind in design['trajectories']
             for name in design['observer_ablation_controllers']]
    order = np.random.default_rng(design['case_order_seed']).permutation(len(plan))
    plan = [plan[i] for i in order]
    for c in plan:
        c['case_id'] = f"{c['mode']}_bias_{'on' if c['bias_enabled'] else 'off'}_{c['trajectory']}_{c['seed']}_{c['controller']}"
    out = args.run_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    for filename in ['plant_parameters.json','motor_model.json','corrected_models.json',
                     'equilibrium_calibration.json','e3_config.json']:
        (out/filename).write_bytes((source/filename).read_bytes())
    write_json(out/'design.json', design); write_json(out/'config.json', settings)
    write_json(out/'planned_cases.json', plan)
    initial = {}
    (out/'initial_conditions').mkdir()
    for seed in design['seeds']:
        p = source/'initial_conditions'/f'{seed}.npz'
        (out/'initial_conditions'/p.name).write_bytes(p.read_bytes())
        with np.load(p) as f: initial[seed] = {k:f[k] for k in f.files}
    sources = [ROOT/p for p in json.loads((source/'FROZEN.json').read_text())['source_sha256']]
    sources += [Path(__file__), args.config.resolve(), ROOT/'pelican_e4_estimation/observation.py',
                ROOT/'pelican_e4_estimation/__init__.py', ROOT/'tests/test_pelican_e4_estimation.py',
                ROOT/'docs/PELICAN_E4_ESTIMATION_DIAGNOSTIC_PROTOCOL.md']
    hashes = {p.relative_to(ROOT).as_posix():checksum(p) for p in sources}
    write_json(out/'FROZEN.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        stage='exploratory_diagnostic', source_sha256=hashes, preserved_files=preserved,
        base_manifest_sha256=checksum(source/'artifact_hashes.json'),
        artifact_sha256={p.relative_to(out).as_posix():checksum(p) for p in out.rglob('*') if p.is_file()},
        python=platform.python_version(), platform=platform.platform(), blas_threads=1))
    controllers = {n:None if n=='pd' else BiasMPC(models[n],motor,parameters,cfg) for n in design['controllers']}
    started = time.monotonic(); rows=[]
    log(out,'started',planned_cases=len(plan))
    for case in plan:
        name,kind,seed = (case[k] for k in ['controller','trajectory','seed'])
        result = run_case(scenario,name,kind,seed,controllers[name],initial[seed],parameters,motor,cfg,
            settings,out/'cases'/case['case_id'],
            lambda event,**v:log(out,event,case_id=case['case_id'],**v),started,
            mode=case['mode'],bias_enabled=case['bias_enabled'],window=design['filter_window_samples'])
        result['case_id']=case['case_id']; rows.append(result)
        pd.DataFrame(rows).to_csv(out/'diagnostic_per_run.csv',index=False)
        log(out,'case_complete',done=len(rows),total=len(plan),**result)
    for p,h in hashes.items(): assert checksum(ROOT/p)==h,p
    preserved_e4(source)
    write_json(out/'COMPLETED.json',dict(cases=len(rows),elapsed_s=time.monotonic()-started,
        complete=sum(r['complete'] for r in rows),tracking_sha256=checksum(out/'diagnostic_per_run.csv')))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/pelican_e4_estimation_diagnostic.json')
    parser.add_argument('--run-dir',type=Path,required=True)
    with threadpool_limits(limits=1): main(parser.parse_args())

