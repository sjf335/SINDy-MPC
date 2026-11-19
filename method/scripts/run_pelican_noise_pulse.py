"""Paired robustness simulations using the frozen Pelican E4 controller and initial states."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pelican_e3_corrected.control import BiasMPC
from run_pelican_e2 import checksum, restore_model, write_json
from run_pelican_e4_estimation_diagnostic import run_case

SOURCE = ROOT / 'runs/pelican_e4_v1'
DIAG = ROOT / 'runs/pelican_e4_estimation_diagnostic_v1'
OUT = ROOT / 'runs/pelican_noise_pulse_v1'
SEEDS = tuple(range(15301,15306))
KINDS = ('circle','figure8')
NAME = 'sindy_history'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def design():
    tasks = []
    for seed in SEEDS:
        for kind in KINDS:
            tasks.extend((case,kind,seed) for case in ('noise_low','noise_mid','pulse_off','pulse_fast'))
        tasks.extend((case,'helix',seed) for case in ('short_on','short_off','short_fast'))
    return tasks


def settings_for(case):
    settings = read(SOURCE/'config.json')
    base = 'pose_noise' if case.startswith('noise') else 'force_pulse'
    scenario = next(s for s in settings['scenarios'] if s['name']==base).copy()
    if case in ('noise_low','noise_mid'):
        scale = {'noise_low':0.1, 'noise_mid':0.5}[case]
        for key in ('position_noise_std_m','rotation_noise_std_rad'):
            scenario[key] *= scale
    if case.endswith('fast'):
        settings['observer']['time_constant_s'] = [0.25]*6
    if case.startswith('short'):
        settings.update(force_end_s=8.5,post_event_start_s=10.5)
    return scenario,settings


def simulate(case,kind,seed):
    case_dir = OUT/'cases'/f'{case}_{kind}_{seed}'
    stamp = case_dir/'manifest.json'
    if stamp.exists():
        for key,digest in read(stamp).items():
            assert checksum(case_dir/key)==digest
        return
    scenario,settings = settings_for(case)
    cfg = read(SOURCE/'e3_config.json')
    if case.startswith('short'):
        cfg = read(ROOT/'runs/pelican_trajectory_search_v1/selected_configs/helix.json')
    params,motor = read(SOURCE/'plant_parameters.json'), read(SOURCE/'motor_model.json')
    model = restore_model(read(SOURCE/'corrected_models.json')[NAME])
    with np.load(SOURCE/'initial_conditions'/f'{seed}.npz') as archive:
        initial = dict(archive)
    run_case(scenario,NAME,kind,seed,BiasMPC(model,motor,params,cfg),initial,params,motor,cfg,
             settings,case_dir,lambda *args,**kwargs:None,time.monotonic(),mode='raw',bias_enabled=not case.endswith('off'))
    write_json(stamp,{name:checksum(case_dir/name) for name in ('control.csv','metrics.json','trajectory.npz')})
    print(case,kind,seed,flush=True)


def main(workers):
    (OUT/'cases').mkdir(parents=True,exist_ok=True)
    tasks = design()
    paths = [SOURCE/name for name in ('config.json','e3_config.json','plant_parameters.json','motor_model.json','corrected_models.json','artifact_hashes.json')]
    paths += [ROOT/'runs/pelican_trajectory_search_v1/selected_configs/helix.json',Path(__file__),ROOT/'scripts/run_pelican_e4_estimation_diagnostic.py',
              ROOT/'pelican_e4/environment.py',ROOT/'pelican_e4_estimation/observation.py',ROOT/'pelican_e3_corrected/control.py']
    paths += [SOURCE/'initial_conditions'/f'{seed}.npz' for seed in SEEDS]
    protocol = dict(tasks=[list(task) for task in tasks],model=NAME,source_hashes={p.relative_to(ROOT).as_posix():checksum(p) for p in paths},
                    noise_fractions=[0,.1,.5,1],tau_s=[.5,.25],short_pulse_s=[8,8.5],long_pulse_s=[8,12],
                    opposite_excursion='maximum reverse signed reference-relative position along pulse direction within 2 s after removal',
                    lag='observed paired pulse-minus-nominal translational residual and estimate in world frame, projected onto pulse axis; first crossing of 50% of injected magnitude',
                    recovery='first 1 s continuously within 3 cm after removal',short_trajectory='paper nominal 1.15 m helix, 0.9 rad/s')
    stamp=OUT/'design.json'
    if stamp.exists(): assert read(stamp)==protocol
    else: write_json(stamp,protocol)
    def worker(task):
        case,kind,seed=task
        subprocess.run([sys.executable,str(Path(__file__)),'--case',case,kind,str(seed)],check=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker,tasks))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',nargs=3)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.case:
        case,kind,seed=args.case
        if (case,kind,int(seed)) not in design(): parser.error('Unknown case')
        simulate(case,kind,int(seed))
    else: main(args.workers)
