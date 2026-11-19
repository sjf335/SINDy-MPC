"""Matched-library negative control for the history ablation, with a fixed training-only shift."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import subprocess
import sys
import time
from run_component_ablation import E2, TRACK, OUT, SEEDS, KINDS, read, write_json, restore_model, checksum, run_case
from predict_component_ablation import bias_history, verify_scalar
import numpy as np
import pandas as pd
from pelican_e2.data import state_features, target_acceleration
from pelican_e2.model import library, fit_candidate, context_for_episode, origins_for, rollout, CURRENT_SIZE
from pelican_e3_corrected.control import calibrate_equilibrium, BiasMPC

CONTROL = OUT/'matched_feature_control'


def fit_shifted_model():
    cfg = read(E2/'config.json')
    base = restore_model(read(E2/'models/sindy_history_h20_a0.01_t0.005.json'))
    train_ids = read(E2/'splits.json')['train']
    scales = {k:base[k] for k in ('mu','scale','y_mu','y_scale')}
    n = base['feature_count']; gram = np.zeros((n,n)); rhs = np.zeros((n,6)); samples = 0
    for fid in train_ids:
        with np.load(E2/'data'/f'flight_{fid:02}.npz') as a:
            e = dict(a)
        idx = np.arange(cfg['training_start_index'],len(e['x'])-1,cfg['fit_stride'])
        z = (np.c_[state_features(e['x'],cfg['gravity']), e['u']] - scales['mu'])/scales['scale']
        phi = library(z[idx],{lag:z[idx-lag] for lag in base['lags']},base['lags'],2)
        # Move the entire 70-column block together, within a training flight only.
        phi[:,CURRENT_SIZE:] = np.roll(phi[:,CURRENT_SIZE:], len(idx)//2, axis=0)
        y = (target_acceleration(e,idx,cfg)-scales['y_mu'])/scales['y_scale']
        weight = 1/len(idx)/len(train_ids)
        gram += weight*(phi.T@phi); rhs += weight*(phi.T@y); samples += len(idx)
    with np.load(E2/'models/training_statistics.npz') as saved:
        for block in (slice(0,CURRENT_SIZE),slice(CURRENT_SIZE,None)):
            np.testing.assert_allclose(gram[block,block],saved['command_gram'][block,block],atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(rhs[:CURRENT_SIZE],saved['command_rhs'][:CURRENT_SIZE],atol=1e-10,rtol=1e-9)
    model = fit_candidate(gram,rhs,scales,base['lags'],2,base['alpha'],base['threshold'],cfg)
    model.update(name='shift_control',family='sindy_history'); assert samples==412555 and len(train_ids)==32
    write_json(CONTROL/'model.json',model)
    write_json(CONTROL/'design.json',dict(train_flights=train_ids,training_samples=samples,features=n,active_coefficients=model['nonzero_coefficients'],
        trace_sha256=checksum(Path(__file__)),model_sha256=checksum(CONTROL/'model.json'),shift='floor(number of training rows / 2) within each flight, all 70 historical differences together'))


def setup(kind):
    params,motor = read(TRACK/'plant_parameters.json'),read(TRACK/'motor_model.json')
    cfg=read(TRACK/'selected_configs'/f'{kind}.json')
    model,_=calibrate_equilibrium(restore_model(read(CONTROL/'model.json')),motor,cfg,params)
    return params,motor,cfg,model


def track(kind,seed):
    dest=CONTROL/'cases'/f'{kind}_{seed}'
    if (dest/'manifest.json').exists():
        for name,digest in read(dest/'manifest.json').items(): assert checksum(dest/name)==digest
        return
    params,motor,cfg,model = setup(kind)
    with np.load(TRACK/'validate/initial_conditions'/f'{seed}.npz') as data: initial=dict(data)
    run_case('combined',kind,'nominal',seed,BiasMPC(model,motor,params,cfg),initial,params,motor,cfg,
             dict(observer=read(TRACK/'observer.json'),disturbance={},stage_runtime_limit_s=3600),dest,
             lambda *args,**kwargs:None,time.monotonic())
    write_json(dest/'manifest.json',{p.name:checksum(p) for p in dest.iterdir() if p.is_file()})
    print('feature control',kind,seed,flush=True)


def score_prediction():
    _,motor,_,model=setup('circle'); cfg=read(E2/'config.json'); obs=read(TRACK/'observer.json'); rows=[]
    for fid in read(E2/'splits.json')['test']:
        with np.load(E2/'data'/f'flight_{fid:02}.npz') as f: e=dict(f)
        origins=origins_for(e,cfg); bias=bias_history(e,model,cfg,obs,20); x,h,prior=context_for_episode(e,origins,model,cfg)
        preds,fail,_=rollout(x,h,e['u'][origins[:,None]+np.arange(50)],prior,{**model,'y_mu':model['y_mu']+bias[origins]},motor,cfg)
        verify_scalar(e,model,cfg,motor,obs,20,origins,bias,preds)
        error=preds[50]-e['x'][origins+50]; ok=fail>50
        rows.append(dict(flight_id=fid,windows=len(ok),failed=int((~ok).sum()),position_m=np.sqrt(np.mean(np.sum(error[ok,:3]**2,axis=1))),
                         velocity_m_s=np.sqrt(np.mean(np.sum(error[ok,3:6]**2,axis=1)))))
    pd.DataFrame(rows).to_csv(CONTROL/'prediction_per_flight.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',nargs=2);parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    if args.case: track(args.case[0],int(args.case[1]))
    else:
        CONTROL.mkdir(exist_ok=True); fit_shifted_model()
        jobs=[(k,s) for s in SEEDS for k in KINDS]
        def run_job(case):
            subprocess.run([sys.executable,__file__,'--case',case[0],str(case[1])],check=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending=[pool.submit(run_job,case) for case in jobs];score_prediction()
            for job in pending: job.result()
