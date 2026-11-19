"""Versioned, paired E3 equilibrium/bias compensation experiment."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from pelican_e3.control import MPC, pack_observation, bounded_command, pd_command
from pelican_e3.plant import observe
from pelican_e3.reference import at
from pelican_e3_corrected.control import BiasMPC, BiasObserver, calibrate_equilibrium, symbolic_transition
from pelican_e3_corrected.plant import disturbance, disturbed_step
from run_pelican_e2 import restore_model, checksum, write_json
from run_pelican_e3 import initialize


def check_originals():
    counts = {}
    for name in ['pelican_e3_v1', 'pelican_e3_diagnostic_v1']:
        path = ROOT/'runs'/name/'artifact_hashes.json'
        sealed = json.loads(path.read_text(encoding='utf-8'))['files']
        for rel, meta in sealed.items():
            if checksum(ROOT/rel) != meta['sha256']:
                raise ValueError('Original sealed artifact changed: '+rel)
        counts[name] = len(sealed)
    return counts


def log(out, event, **values):
    record = dict(utc=datetime.now(timezone.utc).isoformat(), event=event, **values)
    line = json.dumps(record, allow_nan=False)
    with (out/'progress.jsonl').open('a', encoding='utf-8') as f:
        f.write(line+'\n')
    print(line, flush=True)


def run_case(variant, trajectory, scenario, seed, controller, initial, parameters,
             motor, cfg, settings, case_dir, logfn, stage_started):
    kind = trajectory
    states = list(initial['states'].copy()); commands = list(initial['commands'].copy())
    physical = initial['truth'][-1].copy(); prior = initial['motor_prior'].copy()
    previous = commands[-1].copy(); controller.reset()
    enabled = variant in ['online', 'combined']
    transition, _ = symbolic_transition(controller.model, motor, cfg)
    observer = BiasObserver(transition, cfg['dt'], settings['observer'])
    hold = round(cfg['control_dt']/cfg['dt']); nsteps = round(cfg['duration_s']/cfg['dt'])
    poles = np.asarray(motor['poles']); gain, offset = np.asarray(motor['coefficients']).T
    refs = np.array([at(k*cfg['dt'], kind, cfg)[0] for k in range(nsteps+cfg['prediction_steps']+1)])
    truth = [physical.copy()]; measured = [states[-1].copy()]; inputs = []; controls = []
    estimates = [np.zeros(6)]; residuals = []; accepted_log = []; clipped_log = []; applied_bias = []
    reason = 'completed'; started = time.monotonic()
    for k in range(nsteps):
        if time.monotonic()-stage_started > settings['stage_runtime_limit_s']:
            raise TimeoutError('Configured E3 corrected runtime exceeded')
        packed = pack_observation(states, commands, prior, controller.model, cfg)
        if k % hold == 0:
            control_start = time.perf_counter()
            horizon = refs[k+np.arange(1, controller.blocks+1)*hold]
            used = observer.estimate.copy() if enabled else np.zeros(6)
            if variant == 'original':
                requested, stats = controller.solve(packed, horizon, previous)
            else:
                requested, stats = controller.solve(packed, horizon, previous, used)
            fallback = not stats['success']
            if fallback:
                ref, acc = at(k*cfg['dt'], kind, cfg)
                requested = pd_command(states[-1], ref, acc, parameters, cfg)
            u = bounded_command(requested, previous, parameters, cfg)
            controls.append({'step': k, 't': k*cfg['dt'], **stats, 'fallback': fallback,
                             'controller_wall_s': time.perf_counter()-control_start,
                             'bounded_adjustment': float(np.max(np.abs(u-requested))),
                             'max_command_change': float(np.max(np.abs(u-previous))),
                             **{f'u{j+1}': float(u[j]) for j in range(4)},
                             **{f'bias{j}': float(used[j]) for j in range(6)}})
            previous = u.copy()
            if k % 500 == 0:
                logfn('case_progress', variant=variant, trajectory=kind, scenario=scenario,
                      seed=seed, simulated_s=k*cfg['dt'], elapsed_case_s=time.monotonic()-started)
        extra = disturbance(k*cfg['dt'], scenario, settings['disturbance'])
        new = disturbed_step(physical, u, cfg['dt'], parameters, cfg['integration_dt'], extra)
        if not np.isfinite(new).all():
            reason = 'nonfinite_plant'; break
        observation = observe(physical, new, cfg['dt'])
        # k+1 is now available. This update can affect only subsequent solves.
        residual, estimate, accepted, clipped = observer.update(packed, u, observation)
        if not enabled:
            observer.reset(); estimate = np.zeros(6); clipped = False
        residuals.append(residual); estimates.append(estimate)
        accepted_log.append(accepted); clipped_log.append(clipped); applied_bias.append(extra)
        states.append(observation); commands.append(u.copy())
        prior = poles*prior+(1-poles)*(gain*u+offset)
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
    t = np.arange(len(truth))*cfg['dt']; error_vec = truth[:, :3]-reference[:, :3]
    errors = np.linalg.norm(error_vec, axis=1); frame = pd.DataFrame(controls)
    applied = np.asarray(inputs).reshape(-1, 4)
    complete = len(inputs) == nsteps and reason == 'completed'
    bounds = int(np.sum(np.any((applied < np.asarray(parameters['command_lower'])-1e-7)
                               | (applied > np.asarray(parameters['command_upper'])+1e-7), axis=1)))
    slew = int(np.sum(frame.max_command_change > cfg['command_delta_per_control']+1e-7))
    rmse = float(np.sqrt(np.mean(errors**2))); late = t >= cfg['duration_s']/2
    late_mean = error_vec[late].mean(0) if late.any() else np.full(3, np.nan)
    metrics = dict(variant=variant, trajectory=kind, scenario=scenario, seed=seed,
                   complete=complete, termination_reason=reason, duration_completed_s=t[-1],
                   position_rmse_m=rmse if complete else None, prefix_position_rmse_m=rmse,
                   late_rmse_m=float(np.sqrt(np.mean(errors[late]**2))) if late.any() else None,
                   late_bias_norm_m=float(np.linalg.norm(late_mean)) if late.any() else None,
                   max_position_error_m=float(errors.max()), final_position_error_m=float(errors[-1]),
                   solver_failures=int((~frame.success).sum()), fallbacks=int(frame.fallback.sum()),
                   control_calls=len(frame), input_bound_violations=bounds, input_slew_violations=slew,
                   observer_rejected=int(np.sum(~np.asarray(accepted_log))), observer_clipped=int(np.sum(clipped_log)),
                   solve_time_median_s=float(frame.seconds.median()), solve_time_p95_s=float(frame.seconds.quantile(.95)),
                   controller_over_period_fraction=float((frame.controller_wall_s > cfg['control_dt']).mean()),
                   command_variation_squared=float(np.sum(np.diff(np.vstack([initial['commands'][-1], frame[[f'u{i}' for i in range(1, 5)]].to_numpy()]), axis=0)**2)),
                   case_wall_s=time.monotonic()-started)
    case_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(case_dir/'trajectory.npz', t=t, physical=truth, observation=measured,
                        reference=reference, command=applied, bias_estimate=estimates,
                        residual=residuals, accepted=accepted_log, clipped=clipped_log,
                        injected_body_acceleration=applied_bias,
                        initial_history_states=initial['states'], initial_history_commands=initial['commands'])
    frame.to_csv(case_dir/'control.csv', index=False)
    write_json(case_dir/'metrics.json', metrics)
    return metrics


def main(args):
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    stage = settings[args.stage]
    source = ROOT/settings['base_run']
    originals = check_originals()
    cfg = json.loads((source/'config.json').read_text()); cfg['duration_s'] = stage['duration_s']
    parameters = json.loads((source/'plant_parameters.json').read_text())
    motor = json.loads((source/'motor_model.json').read_text())
    model = next(restore_model(m) for m in json.loads((source/'selected_models.json').read_text()) if m['name'] == settings['model'])
    corrected, calibration = calibrate_equilibrium(model, motor, cfg, parameters)
    out = args.run_dir; out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json', settings); write_json(out/'e3_config.json', cfg)
    write_json(out/'plant_parameters.json', parameters); write_json(out/'motor_model.json', motor)
    write_json(out/'original_model.json', model); write_json(out/'corrected_model.json', corrected)
    write_json(out/'equilibrium_calibration.json', calibration)
    plan = [dict(scenario=g['scenario'], trajectory=kind, seed=seed, variant=variant)
            for g in stage['groups'] for kind in g['trajectories'] for seed in stage['seeds'] for variant in g['variants']]
    write_json(out/'planned_cases.json', plan)
    paths = list((ROOT/'pelican_e3_corrected').glob('*.py')) + [Path(__file__), args.config,
            ROOT/'tests/test_pelican_e3_corrected.py']
    source_hashes = {p.relative_to(ROOT).as_posix(): checksum(p) for p in paths}
    if args.stage == 'validation':
        if args.development is None: raise ValueError('Validation requires --development')
        dev = json.loads((args.development/'COMPLETED.json').read_text())
        if not dev['all_complete']: raise ValueError('Development did not complete')
        prior = json.loads((args.development/'FROZEN.json').read_text())
        if prior['source_sha256'] != source_hashes: raise ValueError('Changed implementation since development')
        if json.loads((args.development/'config.json').read_text()) != settings: raise ValueError('Changed settings since development')
        write_json(out/'development_qualification.json', {'path': str(args.development), 'completed_sha256': checksum(args.development/'COMPLETED.json')})
    freeze = dict(created_utc=datetime.now(timezone.utc).isoformat(), stage=args.stage, source_sha256=source_hashes,
                  original_artifacts_verified=originals,
                  artifact_sha256={p.name: checksum(p) for p in out.glob('*.json')},
                  python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'casadi', 'pysindy']})
    write_json(out/'FROZEN.json', freeze)
    started = time.monotonic(); rows = []
    initial = {seed: initialize(seed, parameters, motor, cfg) for seed in stage['seeds']}
    (out/'initial_conditions').mkdir()
    for seed, values in initial.items():
        np.savez_compressed(out/'initial_conditions'/f'{seed}.npz', **values)
    controllers = {}
    for variant in dict.fromkeys(p['variant'] for p in plan):
        chosen = corrected if variant in ['equilibrium', 'combined'] else model
        controllers[variant] = MPC(chosen, motor, parameters, cfg) if variant == 'original' else BiasMPC(chosen, motor, parameters, cfg)
    log(out, 'started', stage=args.stage, planned_cases=len(plan))
    for case in plan:
        name = '{scenario}_{trajectory}_{seed}_{variant}'.format(**case)
        result = run_case(**case, controller=controllers[case['variant']], initial=initial[case['seed']],
                          parameters=parameters, motor=motor, cfg=cfg, settings=settings,
                          case_dir=out/'cases'/name, logfn=lambda event, **v: log(out, event, **v), stage_started=started)
        rows.append(result); pd.DataFrame(rows).to_csv(out/'tracking_per_run.csv', index=False)
        log(out, 'case_complete', done=len(rows), total=len(plan), **result)
    for rel, digest in source_hashes.items():
        if checksum(ROOT/rel) != digest: raise ValueError('Source modified during run: '+rel)
    check_originals()
    write_json(out/'COMPLETED.json', dict(cases=len(rows), all_complete=all(r['complete'] for r in rows),
               fallbacks=sum(r['fallbacks'] for r in rows), elapsed_s=time.monotonic()-started,
               tracking_sha256=checksum(out/'tracking_per_run.csv')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['development', 'validation'], required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/pelican_e3_corrected.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--development', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args)
