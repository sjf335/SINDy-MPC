"""E4 runner: frozen E3 controllers and explicit plant/observation interventions.

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


def check_preserved():
    counts = old_preserved()
    rel = 'runs/pelican_e3_nominal_comparison_v1/artifact_hashes.json'
    files = json.loads((ROOT/rel).read_text(encoding='utf-8'))['files']
    for path, item in files.items():
        assert checksum(ROOT/path) == item['sha256'], path
    counts[rel] = len(files)
    return counts


def run_case(scenario, name, kind, seed, controller, initial, parameters, motor, cfg,
             settings, case_dir, logfn, stage_started):
    noise = noise_samples(seed, len(initial['truth'])+round(cfg['duration_s']/cfg['dt']), scenario, settings)
    observed_initial, previous_pose = observation_history(initial, noise[:len(initial['truth'])], cfg['dt'])
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
            used = observer.estimate.copy() if is_mpc else np.zeros(6)
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
        new_pose = measured_pose(new, noise[len(initial['truth'])+k])
        feedback_start = time.perf_counter()
        observation = observe(previous_pose, new_pose, cfg['dt'])
        previous_pose = new_pose
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
    case_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(case_dir/'trajectory.npz', t=t, physical=truth, observation=measured,
        reference=reference, command=applied, bias_estimate=estimates, residual=residuals,
        accepted=accepted_log, clipped=clipped_log, motor_prior=priors,
        pack_seconds=pack_times, feedback_seconds=feedback_times,
        initial_history_states=observed_initial, initial_history_commands=initial['commands'],
        pose_noise=noise, world_acceleration=acceleration[:len(inputs)])
    frame.to_csv(case_dir/'control.csv', index=False); write_json(case_dir/'metrics.json', metrics)
    return metrics


def main(args):
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    stage = settings[args.stage]; source = ROOT/settings['base_run']
    preserved = check_preserved()
    base_settings = json.loads((source/'config.json').read_text())
    assert settings['observer'] == base_settings['observer']
    assert settings['controllers'] == base_settings['controllers']
    cfg = json.loads((source/'e3_config.json').read_text()); cfg['duration_s'] = stage['duration_s']
    parameters = json.loads((source/'plant_parameters.json').read_text())
    motor = json.loads((source/'motor_model.json').read_text())
    raw_models = json.loads((source/'corrected_models.json').read_text())
    models = {name: restore_model(m) for name, m in raw_models.items()}
    base_frozen = json.loads((source/'FROZEN.json').read_text())
    sources = [ROOT/p for p in base_frozen['source_sha256']]
    sources += [Path(__file__), args.config.resolve(), ROOT/'tests/test_pelican_e4.py',
                ROOT/'docs/PELICAN_E4_PROTOCOL.md', *list((ROOT/'pelican_e4').glob('*.py'))]
    hashes = {p.relative_to(ROOT).as_posix(): checksum(p) for p in sources}
    development = None
    if args.stage == 'formal':
        if args.development is None: raise ValueError('Formal stage requires --development')
        dev = args.development.resolve(); previous = json.loads((dev/'FROZEN.json').read_text())
        assert previous['source_sha256'] == hashes, 'Changed since development'
        audit = json.loads((dev/'verification.json').read_text())
        assert audit['status'] == 'PASS'
        assert audit['tracking_sha256'] == checksum(dev/'robustness_per_run.csv')
        development = dict(path=dev.relative_to(ROOT).as_posix(), verification_sha256=checksum(dev/'verification.json'))
    out = args.run_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    # Exact artifact copies make the absence of recalibration/refitting auditable.
    for filename in ['plant_parameters.json', 'motor_model.json', 'corrected_models.json', 'equilibrium_calibration.json']:
        (out/filename).write_bytes((source/filename).read_bytes())
    write_json(out/'config.json', settings); write_json(out/'e3_config.json', cfg)
    rng = np.random.default_rng(settings['case_order_seed'])
    plan = [dict(scenario=s['name'], controller=name, trajectory=kind, seed=seed)
            for s in settings['scenarios'] for kind in stage['trajectories'] for seed in stage['seeds']
            for name in rng.permutation(settings['controllers'])]
    write_json(out/'planned_cases.json', plan)
    initial = {seed: initialize(seed, parameters, motor, cfg) for seed in stage['seeds']}
    (out/'initial_conditions').mkdir()
    for seed, values in initial.items(): np.savez_compressed(out/'initial_conditions'/f'{seed}.npz', **values)
    write_json(out/'FROZEN.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), stage=args.stage,
        source_sha256=hashes, artifact_sha256={p.relative_to(out).as_posix(): checksum(p) for p in out.rglob('*') if p.is_file()},
        base_manifest_sha256=checksum(source/'artifact_hashes.json'), preserved_files=preserved, development=development,
        python=platform.python_version(), platform=platform.platform(), processor=platform.processor(),
        logical_cpu_count=os.cpu_count(), blas_threads=1,
        packages={p: importlib.metadata.version(p) for p in ['numpy','scipy','casadi','pandas','matplotlib']}))
    controllers = {name: None if name == 'pd' else BiasMPC(models[name], motor, parameters, cfg) for name in settings['controllers']}
    scenarios = {s['name']: s for s in settings['scenarios']}
    started = time.monotonic(); rows = []
    log(out, 'started', stage=args.stage, planned_cases=len(plan))
    for case in plan:
        scenario, name, kind, seed = (case[k] for k in ['scenario','controller','trajectory','seed'])
        result = run_case(scenarios[scenario], name, kind, seed, controllers[name], initial[seed], parameters,
            motor, cfg, settings, out/'cases'/f'{scenario}_{kind}_{seed}_{name}',
            lambda event, **v: log(out, event, scenario=scenario, **v), started)
        rows.append(result); pd.DataFrame(rows).to_csv(out/'robustness_per_run.csv', index=False)
        log(out, 'case_complete', done=len(rows), total=len(plan), **result)
    for rel, value in hashes.items(): assert checksum(ROOT/rel) == value, rel
    check_preserved()
    write_json(out/'COMPLETED.json', dict(cases=len(rows), all_complete=all(r['complete'] for r in rows),
        fallbacks=sum(r['fallbacks'] for r in rows), elapsed_s=time.monotonic()-started,
        tracking_sha256=checksum(out/'robustness_per_run.csv')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['development','formal'], required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/pelican_e4.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--development', type=Path)
    with threadpool_limits(limits=1): main(parser.parse_args())
