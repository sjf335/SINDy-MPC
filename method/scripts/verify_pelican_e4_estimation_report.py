"""Check aggregation and fixed-path sensitivity using the CasADi transition.

The report uses NumPy library evaluation and direct least squares; this audit
uses the runtime causal estimator and mapped symbolic dynamics instead.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from pelican_e4_estimation.observation import CausalPolynomialPose
from pelican_e3.control import symbolic_transition
from run_pelican_e2 import restore_model,write_json,checksum
from report_pelican_e4_estimation_diagnostic import load_npz,pose_observations,observation_metrics


def verify(out):
    out=out.resolve();design=json.loads((out/'design.json').read_text());source=ROOT/design['base_run']
    cfg=json.loads((out/'e3_config.json').read_text());dt=cfg['dt']
    settings=json.loads((out/'config.json').read_text());gate=np.asarray(settings['observer']['residual_gate'])
    motor=json.loads((out/'motor_model.json').read_text())
    models={k:restore_model(v) for k,v in json.loads((out/'corrected_models.json').read_text()).items()}
    replay=pd.read_csv(out/'fixed_path_replay.csv');summary=pd.read_csv(out/'diagnostic_summary.csv')
    combined=pd.read_csv(out/'paired_all_runs.csv')
    assert len(combined)==88 and len(replay)==140
    for row in summary.itertuples(index=False):
        g=combined[(combined['mode']==row.mode)&(combined.bias_enabled==row.bias_enabled)&(combined.controller==row.controller)]
        assert len(g)==row.cases and int(g.complete.sum())==row.complete
        if g.complete.all():assert_allclose(g.position_rmse_m.mean(),row.position_rmse_m,atol=1e-12)
        else:assert pd.isna(row.position_rmse_m)
        assert_allclose(g.max_position_error_m.max(),row.max_position_error_m,atol=1e-12)
        periods=pd.concat([pd.read_csv(ROOT/p/'control.csv') for p in g.case_path])
        assert_allclose(periods.full_control_period_s.quantile(.95),row.full_period_p95_s,atol=1e-12)
        assert_allclose((periods.full_control_period_s>cfg['control_dt']).mean(),row.over_period_fraction,atol=1e-12)
    rows=0;targets=0;max_error=0.
    for seed in design['seeds']:
        init=load_npz(source/'initial_conditions'/f'{seed}.npz');n0=len(init['commands'])
        for kind in design['trajectories']:
            for name in design['controllers']:
                d=load_npz(source/'cases'/f'nominal_{kind}_{seed}_{name}'/'trajectory.npz')
                truth=np.vstack([init['truth'],d['physical'][1:]])
                noise=np.random.default_rng(settings['noise_seed_offset']+seed).standard_normal((len(truth),6))*.001
                clean,raw=pose_observations(truth,noise,dt)
                filt=CausalPolynomialPose(dt,design['filter_window_samples'])
                filtered=np.array([filt.update(x[:3],x[6:10]) for x in raw])
                oracle=raw.copy();oracle[:,3:6]=clean[:,3:6];oracle[:,10:13]=clean[:,10:13]
                modes=dict(nominal=clean,raw=raw,oracle_rates=oracle,causal_poly=filtered)
                for label,cols in [('velocity_only',list(range(3,6))),('rate_only',list(range(10,13))),('pose_only',[0,1,2,6,7,8,9])]:
                    x=clean.copy();x[:,cols]=raw[:,cols];modes[label]=x
                commands=np.vstack([init['commands'],d['command']]);n=len(d['command']);idx=n0+np.arange(n)
                if name!='pd':
                    model=models[name];fn,_=symbolic_transition(model,motor,cfg);mapped=fn.map(n)
                for mode,x in modes.items():
                    row=replay[(replay['mode']==mode)&(replay.controller==name)&(replay.trajectory==kind)&(replay.seed==seed)].iloc[0]
                    for key,v in observation_metrics(x[n0:],clean[n0:],truth[n0:],d['t']).items():assert_allclose(row[key],v,atol=2e-11)
                    if name!='pd':
                        pieces=[x[idx]];lagmax=max(model['lags'],default=0)
                        if lagmax:
                            inv=Rotation.from_quat(x[:-1,6:10]).inv()
                            raw_features=np.c_[inv.apply(x[:-1,3:6]),x[:-1,10:13],inv.apply(np.tile([0,0,-cfg['gravity']],(len(commands),1))),commands]
                            z=(raw_features-model['mu'])/model['scale']
                            pieces.append(np.stack([z[idx-lag] for lag in range(1,lagmax+1)],axis=1).reshape(n,-1))
                        if model['motor']:pieces.append(d['motor_prior'][:-1])
                        packed=np.hstack(pieces)
                        pred=np.asarray(mapped(packed.T,d['command'].T)).T[:,:13]
                        target=np.c_[Rotation.from_quat(x[idx,6:10]).inv().apply((pred[:,3:6]-x[idx,3:6])/dt+[0,0,cfg['gravity']]),
                                     (pred[:,10:13]-x[idx,10:13])/dt]
                        if mode=='nominal':base=target
                        perturb=target-base
                        for label,part in [('predicted_force_perturbation_rms',slice(0,3)),('predicted_angular_acceleration_perturbation_rms',slice(3,6))]:
                            v=float(np.sqrt(np.mean(np.sum(perturb[:,part]**2,axis=1))))
                            assert_allclose(row[label],v,atol=2e-10,rtol=1e-10);max_error=max(max_error,abs(row[label]-v))
                        residual=np.c_[Rotation.from_quat(x[idx,6:10]).inv().apply((x[idx+1,3:6]-pred[:,3:6])/dt),(x[idx+1,10:13]-pred[:,10:13])/dt]
                        assert_allclose(np.any(np.abs(residual)>gate,axis=1).mean(),row.observer_rejection_fraction,atol=1e-12)
                        targets+=n
                    rows+=1
    write_json(out/'report_verification.json',dict(status='PASS',combined_cases=len(combined),summary_rows=len(summary),
        fixed_path_rows=rows,symbolic_target_evaluations=targets,max_sensitivity_metric_error=max_error,
        inputs_sha256={f:checksum(out/f) for f in ['paired_all_runs.csv','diagnostic_summary.csv','fixed_path_replay.csv','fixed_path_replay_summary.csv']},
        script_sha256=checksum(Path(__file__))))
    print('PASS:',rows,'fixed-path rows;',targets,'symbolic target evaluations; max metric error',max_error)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    with threadpool_limits(limits=1):verify(p.parse_args().run_dir)
