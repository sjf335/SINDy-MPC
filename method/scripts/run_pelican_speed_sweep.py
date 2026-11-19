"""Reference-speed sensitivity using the sealed E3 control loop and models."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pelican_e3_corrected.control import BiasMPC
from run_pelican_e2 import checksum, restore_model, write_json
from run_pelican_e3 import initialize
from run_pelican_e3_nominal_comparison import run_case, log
from run_pelican_e4 import check_preserved as check_e3

SNAPSHOTS = ('plant_parameters.json', 'motor_model.json', 'corrected_models.json',
             'equilibrium_calibration.json')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_preserved():
    counts = check_e3()
    rel = 'runs/pelican_e4_v1/artifact_hashes.json'
    files = read_json(ROOT / rel)['files']
    for path, item in files.items():
        assert checksum(ROOT / path) == item['sha256'], path
    counts[rel] = len(files)
    return counts


def case_config(base, frequency, duration):
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError('Reference frequency must be positive and finite')
    cfg = deepcopy(base)
    cfg['trajectory_frequency_rad_s'] = float(frequency)
    cfg['duration_s'] = float(duration)
    return cfg


def activity_metrics(folder, params, initial, cfg):
    frame = pd.read_csv(folder / 'control.csv')
    commands = frame[[f'u{j}' for j in range(1, 5)]].to_numpy()
    delta = np.diff(np.vstack([initial['commands'][-1], commands]), axis=0)
    lower, upper = np.asarray(params['command_lower']), np.asarray(params['command_upper'])
    return dict(
        input_bound_active_fraction=float(np.mean(np.any(
            (commands <= lower + 1e-5) | (commands >= upper - 1e-5), axis=1))),
        input_slew_active_fraction=float(np.mean(np.any(
            np.abs(delta) >= cfg['command_delta_per_control'] - 1e-5, axis=1))),
        command_limiter_active_fraction=float(np.mean(frame.bounded_adjustment > 1e-7)))


def main(args):
    settings = read_json(args.config)
    source = ROOT / settings['base_run']
    base_settings = read_json(source / 'config.json')
    base = read_json(source / 'e3_config.json')
    stage = settings[args.stage]
    assert settings['controllers'] == base_settings['controllers']
    assert not set(settings['development']['seeds']) & set(settings['formal']['seeds'])
    assert len(set(settings['frequency_rad_s'])) == len(settings['frequency_rad_s'])
    preserved = check_preserved()
    params, motor = (read_json(source / name) for name in SNAPSHOTS[:2])
    models = {name: restore_model(model) for name, model in
              read_json(source / 'corrected_models.json').items()}
    frozen = read_json(source / 'FROZEN.json')
    paths = [ROOT / p for p in frozen['source_sha256']]
    paths += [Path(__file__), args.config.resolve(), ROOT / 'docs/PELICAN_SPEED_SWEEP_PROTOCOL.md',
              ROOT / 'scripts/verify_pelican_speed_sweep.py', ROOT / 'scripts/report_pelican_speed_sweep.py',
              ROOT / 'tests/test_pelican_speed_sweep.py', ROOT / 'scripts/run_pelican_e4.py']
    hashes = {p.relative_to(ROOT).as_posix(): checksum(p) for p in paths}
    development = None
    if args.stage == 'formal':
        if args.development is None:
            raise ValueError('Formal stage requires a verified --development run')
        dev = args.development.resolve()
        audit = read_json(dev / 'verification.json')
        assert read_json(dev / 'FROZEN.json')['source_sha256'] == hashes
        assert read_json(dev / 'config.json') == settings
        assert audit['status'] == 'PASS'
        assert audit['tracking_sha256'] == checksum(dev / 'tracking_per_run.csv')
        development = dict(path=dev.relative_to(ROOT).as_posix(),
                           verification_sha256=checksum(dev / 'verification.json'))
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in SNAPSHOTS:
        (out / name).write_bytes((source / name).read_bytes())
    write_json(out / 'config.json', settings)
    write_json(out / 'base_e3_config.json', base)
    write_json(out / 'observer.json', base_settings['observer'])
    rng = np.random.default_rng(settings['case_order_seed'])
    plan = [dict(speed_index=i, frequency_rad_s=w, trajectory=kind, seed=seed, controller=name)
            for i, w in enumerate(settings['frequency_rad_s'])
            for kind in stage['trajectories'] for seed in stage['seeds']
            for name in rng.permutation(settings['controllers'])]
    for item in plan:
        item['case_id'] = f"speed{item['speed_index']}_{item['trajectory']}_{item['seed']}_{item['controller']}"
    write_json(out / 'planned_cases.json', plan)
    (out / 'reference_configs').mkdir()
    for i, w in enumerate(settings['frequency_rad_s']):
        write_json(out / 'reference_configs' / f'speed{i}.json', case_config(base, w, stage['duration_s']))
    initial = {seed: initialize(seed, params, motor, base) for seed in stage['seeds']}
    (out / 'initial_conditions').mkdir()
    for seed, values in initial.items():
        np.savez_compressed(out / 'initial_conditions' / f'{seed}.npz', **values)
    write_json(out / 'FROZEN.json', dict(stage=args.stage, source_sha256=hashes,
        created_utc=datetime.now(timezone.utc).isoformat(), development=development,
        preserved_files=preserved, python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'casadi', 'pysindy']},
        artifact_sha256={p.relative_to(out).as_posix(): checksum(p) for p in out.rglob('*') if p.is_file()}))
    started = time.monotonic()
    rows, controllers, previous_speed = [], {}, None
    runtime = dict(observer=base_settings['observer'], stage_runtime_limit_s=settings['stage_runtime_limit_s'])
    log(out, 'started', stage=args.stage, planned_cases=len(plan))
    for case in plan:
        cfg = read_json(out / 'reference_configs' / f"speed{case['speed_index']}.json")
        if previous_speed != case['speed_index']:
            controllers = {name: None if name == 'pd' else BiasMPC(models[name], motor, params, cfg)
                           for name in settings['controllers']}
            previous_speed = case['speed_index']
        folder = out / 'cases' / case['case_id']
        result = run_case(case['controller'], case['trajectory'], case['seed'],
            controllers[case['controller']], initial[case['seed']], params, motor, cfg,
            runtime, folder, lambda event, **v: log(out, event, frequency_rad_s=case['frequency_rad_s'], **v), started)
        result.update(case)
        result.update(activity_metrics(folder, params, initial[case['seed']], cfg))
        write_json(folder / 'metrics.json', result)
        rows.append(result)
        pd.DataFrame(rows).to_csv(out / 'tracking_per_run.csv', index=False)
        log(out, 'case_complete', done=len(rows), total=len(plan), **result)
    for rel, value in hashes.items():
        assert checksum(ROOT / rel) == value, rel
    check_preserved()
    write_json(out / 'COMPLETED.json', dict(cases=len(rows),
        all_complete=all(r['complete'] for r in rows), elapsed_s=time.monotonic() - started,
        tracking_sha256=checksum(out / 'tracking_per_run.csv')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pelican_speed_sweep.json')
    parser.add_argument('--stage', choices=['development', 'formal'], required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--development', type=Path)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
