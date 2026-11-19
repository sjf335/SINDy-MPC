"""Adaptive diagnostic: separate force and angular-acceleration intercept corrections."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from diagnose_pelican_e3 import load,log,static_correction
from pelican_e3.control import MPC
from run_pelican_e3 import initialize,run_case
from run_pelican_e2 import checksum,write_json


def main(out):
    cfg,p,motor,models=load()
    assert (out/'intervention/COMPLETED.json').exists()
    target=out/'bias_components';target.mkdir(exist_ok=False)
    write_json(target/'ADAPTIVE_PROTOCOL.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        reason='The first paired history case improved markedly after six-output intercept calibration. Separate force vs angular contributions.',
        variants={'force_only':'subtract only first 3 hover acceleration residuals',
                  'angular_only':'subtract only final 3 hover angular acceleration residuals'},
        models=['sindy_history','motor_sindy'],seeds=[9301,9302],trajectories=['circle','figure8'],
        cases=16,original_and_full_correction_reference='intervention/per_run.csv',
        source_script_sha256=checksum(Path(__file__)),exploratory=True))
    start=time.monotonic();seeds=[9301,9302]
    initial={seed:initialize(seed,p,motor,cfg) for seed in seeds}
    (target/'initial_conditions').mkdir()
    for seed,value in initial.items():np.savez_compressed(target/'initial_conditions'/f'{seed}.npz',**value)
    controllers={}
    for name in ['sindy_history','motor_sindy']:
        model=models[name];_,residual=static_correction(model,motor,cfg,p)
        for variant,mask in [('force_only',[1,1,1,0,0,0]),('angular_only',[0,0,0,1,1,1])]:
            modified={**model,'y_mu':model['y_mu']-residual*np.array(mask)}
            controllers[(name,variant)]=MPC(modified,motor,p,cfg)
    records=[]
    def logger(event,**values):log(out,event,**values)
    for kind in ['circle','figure8']:
        for seed in seeds:
            for name in ['sindy_history','motor_sindy']:
                for variant in ['force_only','angular_only']:
                    log(out,'component_started',model=name,variant=variant,trajectory=kind,seed=seed)
                    m=run_case(name,kind,seed,controllers[(name,variant)],initial[seed],p,motor,cfg,
                               target/'cases'/f'{kind}_{seed}_{name}_{variant}',logger,start)
                    records.append({**m,'variant':variant});pd.DataFrame(records).to_csv(target/'per_run.csv',index=False)
                    log(out,'component_completed',variant=variant,**m)
    write_json(target/'COMPLETED.json',dict(cases=len(records),elapsed_s=time.monotonic()-start))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/pelican_e3_diagnostic_v1')
    with threadpool_limits(limits=1):main(parser.parse_args().run_dir)
