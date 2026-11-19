"""Five prespecified ablations on the manuscript's frozen flight and tracking splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import json
import os
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from pelican_e3_corrected.control import BiasMPC, calibrate_equilibrium
from run_pelican_e2 import checksum, restore_model, write_json
from run_pelican_e3_corrected import run_case

E2 = ROOT / 'runs/pelican_e2_v1'
TRACK = ROOT / 'runs/pelican_trajectory_search_v1'
OUT = ROOT / 'runs/pelican_component_ablation_v1'
# history features, equilibrium correction, online residual
GROUPS = {'full': (True, True, True), 'no_history': (False, True, True),
          'no_calibration': (True, False, True), 'no_bias': (True, True, False),
          'none': (False, False, False)}
SEEDS = tuple(range(24301, 24306))
KINDS = ('circle', 'figure8', 'helix')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def get_model(group, cfg):
    history, calibration, online = GROUPS[group]
    stem = 'sindy_history_h20' if history else 'sindy_current_h0'
    model = restore_model(read(E2 / 'models' / (stem + '_a0.01_t0.005.json')))
    model['name'] = model['family']
    if calibration:
        model, _ = calibrate_equilibrium(model, read(TRACK/'motor_model.json'), cfg, read(TRACK/'plant_parameters.json'))
    return model


def run_one(group, kind, seed):
    folder = OUT / 'cases' / f'{group}_{kind}_{seed}'
    cfg = read(TRACK/'selected_configs'/f'{kind}.json')
    params, motor = read(TRACK/'plant_parameters.json'), read(TRACK/'motor_model.json')
    settings = dict(observer=read(TRACK/'observer.json'), disturbance={}, stage_runtime_limit_s=3600)
    with np.load(TRACK/'validate/initial_conditions'/f'{seed}.npz') as f:
        initial = dict(f)
    if not (folder/'manifest.json').exists():
        controller = BiasMPC(get_model(group, cfg), motor, params, cfg)
        mode = {'no_history': 'combined', 'no_calibration': 'online', 'no_bias': 'equilibrium', 'none': 'original'}[group]
        run_case(mode, kind, 'nominal', seed, controller, initial, params, motor, cfg, settings,
                 folder, lambda *args, **kwargs: None, time.monotonic())
        write_json(folder/'manifest.json', {f.name:checksum(f) for f in folder.iterdir() if f.is_file()})
    for name, digest in read(folder/'manifest.json').items():
        assert checksum(folder/name) == digest
    print(f'{group} {kind} {seed} completed', flush=True)


def tracking(workers):
    OUT.mkdir(exist_ok=True)
    cfg = read(TRACK/'config.json')
    assert tuple(cfg['validation_seeds']) == SEEDS and tuple(cfg['trajectories']) == KINDS
    assert read(TRACK/'validate/verification.json')['status'] == 'PASS'
    source_files = [TRACK/'observer.json', TRACK/'plant_parameters.json', TRACK/'motor_model.json', TRACK/'corrected_models.json',
                    E2/'splits.json', E2/'config.json', Path(__file__), ROOT/'scripts/run_pelican_e3_corrected.py',
                    ROOT/'pelican_e3_corrected/control.py', ROOT/'pelican_e3_corrected/plant.py', ROOT/'pelican_e3/control.py',
                    ROOT/'pelican_e3/plant.py',ROOT/'pelican_e3/reference.py', ROOT/'pelican_e2/data.py', ROOT/'pelican_e2/model.py']
    source_files.extend(TRACK/'selected_configs'/f'{k}.json' for k in KINDS)
    source_files.extend(TRACK/'validate/initial_conditions'/f'{s}.npz' for s in SEEDS)
    source_files.extend(E2/'models'/f'{stem}_a0.01_t0.005.json' for stem in ('sindy_history_h20','sindy_current_h0'))
    freeze = dict(groups=GROUPS, seeds=SEEDS, trajectories=KINDS, test_flights=read(E2/'splits.json')['test'],
                  sources={p.relative_to(ROOT).as_posix():checksum(p) for p in source_files})
    record = OUT/'design.json'
    if record.exists():
        assert read(record) == json.loads(json.dumps(freeze)), 'Run inputs have changed'
    else:
        write_json(record, freeze)
    for k in ('coefficients','y_mu','mu','scale','y_scale','feature_scale'):
        np.testing.assert_array_equal(get_model('full',read(TRACK/'selected_configs/circle.json'))[k], read(TRACK/'corrected_models.json')['sindy_history'][k])
    tasks = [(g,k,s) for s in SEEDS for k in KINDS for g in GROUPS if g != 'full']
    def run_task(task):
        g,k,s = task
        subprocess.run([sys.executable, str(Path(__file__)), '--case',g,k,str(s)], check=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_task,tasks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', nargs=3)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.case:
        group,kind,seed = args.case
        run_one(group,kind,int(seed))
    else:
        tracking(args.workers)
