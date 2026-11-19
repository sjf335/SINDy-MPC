"""Bounded, auditable reference search followed by untouched-initial-state validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from threadpoolctl import threadpool_limits
from pelican_e3_corrected.control import BiasMPC
from run_pelican_e2 import checksum, restore_model, write_json
from run_pelican_e3 import initialize
from run_pelican_e3_nominal_comparison import run_case
from run_pelican_speed_sweep import activity_metrics
from pelican_trajectory_search_common import (read_json, check_preserved, SNAPSHOTS, ORDER,
    configuration, stage_design)


def block_worker(task):
    """One paired candidate/seed block; a process owns every output file in it."""
    with threadpool_limits(limits=1):
        root, out = Path(task['root']), Path(task['out'])
        settings = read_json(root / 'config.json')
        cfg = read_json(out / 'reference_configs' / (task['candidate']['candidate_id'] + '.json'))
        params, motor = (read_json(root / name) for name in SNAPSHOTS[:2])
        models = {name: restore_model(m) for name, m in read_json(root / 'corrected_models.json').items()}
        with np.load(out / 'initial_conditions' / f"{task['seed']}.npz") as f:
            initial = {k: f[k] for k in f.files}
        runtime = dict(observer=read_json(root / 'observer.json'), stage_runtime_limit_s=settings['stage_runtime_limit_s'])
        result_rows = []
        progress = out / 'block_progress' / f"{task['candidate']['candidate_id']}_{task['seed']}.json"
        for case in task['cases']:
            folder = out / 'cases' / case['case_id']
            if (folder / 'case_manifest.json').exists():
                manifest = read_json(folder / 'case_manifest.json')
                for name, value in manifest.items():
                    assert checksum(folder / name) == value, folder / name
                metrics = read_json(folder / 'metrics.json')
                assert all(metrics[k] == v for k, v in case.items())
                result_rows.append(metrics)
                continue
            if folder.exists():
                raise ValueError(f'Incomplete case directory retained for inspection: {folder}')
            reused = task['reuse'].get(case['controller'])
            if reused:
                original = ROOT / reused
                folder.mkdir(parents=True)
                for name in ['trajectory.npz', 'control.csv']:
                    shutil.copyfile(original / name, folder / name)
                with np.load(original / 'trajectory.npz') as d:
                    assert_array_equal(d['initial_history_states'], initial['states'])
                    assert_array_equal(d['initial_history_commands'], initial['commands'])
                    assert_array_equal(d['physical'][0], initial['truth'][-1])
                old = read_json(original / 'metrics.json')
                metrics = {k: v for k, v in old.items() if k not in ['case_id', 'speed_index', 'frequency_rad_s']}
                write_json(folder / 'reuse_source.json', dict(path=reused,
                    files={name: checksum(original / name) for name in ['trajectory.npz', 'control.csv', 'metrics.json']}))
            else:
                controller = None if case['controller'] == 'pd' else BiasMPC(models[case['controller']], motor, params, cfg)
                def progress_log(event, **values):
                    write_json(progress, dict(event=event, **case, **{k: v for k, v in values.items() if k not in case}))
                metrics = run_case(case['controller'], case['trajectory'], case['seed'], controller,
                    initial, params, motor, cfg, runtime, folder, progress_log, task['started'])
            metrics.update(case)
            metrics.update(activity_metrics(folder, params, initial, cfg))
            metrics['reused'] = bool(reused)
            write_json(folder / 'metrics.json', metrics)
            write_json(folder / 'case_manifest.json', {p.name: checksum(p) for p in folder.iterdir() if p.is_file()})
            result_rows.append(metrics)
            write_json(progress, dict(event='case_complete', **case, complete=metrics['complete'],
                position_rmse_m=metrics['position_rmse_m'], done_in_block=len(result_rows)))
        return result_rows


def freeze_root(root, config):
    settings = read_json(config)
    source = ROOT / settings['base_run']
    assert settings['controllers'] == ORDER == read_json(source / 'config.json')['controllers']
    seed_groups = [set(settings[k]) for k in ['development_seeds', 'confirmation_seeds', 'validation_seeds']]
    assert not (seed_groups[0] & seed_groups[1] or seed_groups[0] & seed_groups[2] or seed_groups[1] & seed_groups[2])
    previous = read_json(source / 'FROZEN.json')
    paths = [ROOT / p for p in previous['source_sha256']]
    paths += [Path(__file__), config.resolve(), ROOT / 'scripts/pelican_trajectory_search_common.py',
        ROOT / 'scripts/verify_pelican_trajectory_search.py', ROOT / 'scripts/run_pelican_speed_sweep.py',
        ROOT / 'scripts/run_pelican_e4.py', ROOT / 'docs/PELICAN_TRAJECTORY_SEARCH_PROTOCOL.md',
        ROOT / 'tests/test_pelican_trajectory_search.py']
    hashes = {p.relative_to(ROOT).as_posix(): checksum(p) for p in paths}
    if root.exists():
        frozen = read_json(root / 'FROZEN.json')
        assert read_json(root / 'config.json') == settings
        assert frozen['source_sha256'] == hashes, 'Search code or plan changed; use a new version'
        for name, value in frozen['artifact_sha256'].items():
            assert checksum(root / name) == value
        return settings
    preserved = check_preserved()
    root.mkdir(parents=True, exist_ok=False)
    for name in SNAPSHOTS:
        shutil.copyfile(source / name, root / name)
    write_json(root / 'config.json', settings)
    write_json(root / 'base_e3_config.json', read_json(source / 'e3_config.json'))
    write_json(root / 'observer.json', read_json(source / 'config.json')['observer'])
    write_json(root / 'FROZEN.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashes, preserved_files=preserved, python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'casadi', 'pysindy']},
        artifact_sha256={p.name: checksum(p) for p in root.iterdir() if p.is_file()}))
    return settings


def main(args):
    root = args.run_dir.resolve()
    settings = freeze_root(root, args.config)
    candidates, seeds, ranking = stage_design(root, args.stage, settings)
    out = root / args.stage
    base = read_json(root / 'base_e3_config.json')
    params, motor = (read_json(root / name) for name in SNAPSHOTS[:2])
    rng = np.random.default_rng(settings['case_order_seed'])
    plan = []
    for c in candidates:
        for seed in seeds:
            for name in rng.permutation(ORDER):
                plan.append(dict(**c, seed=seed, controller=name, case_id=f"{c['candidate_id']}_{seed}_{name}"))
    if out.exists():
        assert read_json(out / 'candidates.json') == candidates
        assert read_json(out / 'planned_cases.json') == plan
        for name, value in read_json(out / 'FROZEN.json')['artifact_sha256'].items():
            assert checksum(out / name) == value
        if (out / 'COMPLETED.json').exists():
            print(json.dumps(dict(stage=args.stage, already_completed=True)))
            return
    else:
        out.mkdir()
        write_json(out / 'candidates.json', candidates)
        write_json(out / 'planned_cases.json', plan)
        write_json(out / 'stage.json', dict(stage=args.stage, seeds=seeds, duration_s=settings['duration_s']))
        if ranking is not None:
            ranking.to_csv(out / 'input_development_ranking.csv', index=False)
        if args.stage == 'validate':
            selected = dict(selection_stage='development_only', candidates=candidates,
                development_ranking_sha256=checksum(out / 'input_development_ranking.csv'),
                created_utc=datetime.now(timezone.utc).isoformat(),
                validation_seeds=settings['validation_seeds'])
            write_json(root / 'selection.json', selected)
            write_json(out / 'selection.json', selected)
        (out / 'reference_configs').mkdir()
        for c in candidates:
            write_json(out / 'reference_configs' / (c['candidate_id'] + '.json'), configuration(base, c, settings['duration_s']))
        (out / 'initial_conditions').mkdir()
        for seed in seeds:
            np.savez_compressed(out / 'initial_conditions' / f'{seed}.npz', **initialize(seed, params, motor, base))
        (out / 'block_progress').mkdir()
        before = {'grid': [], 'refine': ['grid'], 'confirm': ['grid', 'refine'], 'validate': ['grid', 'refine', 'confirm']}[args.stage]
        write_json(out / 'FROZEN.json', dict(stage=args.stage,
            parent_frozen_sha256=checksum(root / 'FROZEN.json'),
            created_utc=datetime.now(timezone.utc).isoformat(),
            prior_audits={s: checksum(root / s / 'verification.json') for s in before},
            artifact_sha256={p.relative_to(out).as_posix(): checksum(p) for p in out.rglob('*') if p.is_file()}))
    started = time.monotonic()
    oldrun = ROOT / settings['reuse_run']
    oldcases = pd.read_csv(oldrun / 'tracking_per_run.csv')
    oldcfg = read_json(oldrun / 'base_e3_config.json')
    tasks = []
    for c in candidates:
        for seed in seeds:
            reuse = {}
            if args.stage == 'grid' and c['trajectory'] == 'figure8' and c['radius_m'] == oldcfg['trajectory_radius_m']:
                subset = oldcases[(oldcases.frequency_rad_s == c['frequency_rad_s']) & (oldcases.seed == seed)]
                for row in subset.itertuples(index=False):
                    check_cfg = read_json(oldrun / 'reference_configs' / f'speed{row.speed_index}.json')
                    assert check_cfg == configuration(base, c, settings['duration_s'])
                    reuse[row.controller] = (oldrun / 'cases' / row.case_id).relative_to(ROOT).as_posix()
            tasks.append(dict(root=str(root), out=str(out), candidate=c, seed=seed, started=started, reuse=reuse,
                cases=[p for p in plan if p['candidate_id'] == c['candidate_id'] and p['seed'] == seed]))
    print(json.dumps(dict(event='started', stage=args.stage, candidates=len(candidates), cases=len(plan), workers=settings['workers'])), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=settings['workers']) as pool:
        futures = [pool.submit(block_worker, task) for task in tasks]
        for future in as_completed(futures):
            rows.extend(future.result())
            pd.DataFrame(rows).sort_values('case_id').to_csv(out / 'tracking_per_run.csv', index=False)
            event = dict(event='block_complete', stage=args.stage, done=len(rows), total=len(plan),
                         elapsed_s=round(time.monotonic() - started, 1))
            with (out / 'progress.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')
            print(json.dumps(event), flush=True)
    for rel, value in read_json(root / 'FROZEN.json')['source_sha256'].items():
        assert checksum(ROOT / rel) == value
    check_preserved()
    write_json(out / 'COMPLETED.json', dict(cases=len(rows), completed=sum(r['complete'] for r in rows),
        reused=sum(r['reused'] for r in rows), elapsed_s=time.monotonic() - started,
        tracking_sha256=checksum(out / 'tracking_per_run.csv')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pelican_trajectory_search.json')
    parser.add_argument('--stage', choices=['grid', 'refine', 'confirm', 'validate'], required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
