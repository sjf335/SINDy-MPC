"""Post-E3 exploratory replay, input-response and equilibrium-intervention diagnosis."""
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
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from pelican_e2.data import state_features,target_acceleration,rotate
from pelican_e2.model import context_for_episode,rollout
from pelican_e3.plant import step,derivative
from pelican_e3.control import symbolic_transition,pack_observation,MPC
from run_pelican_e2 import restore_model,checksum,write_json
from run_pelican_e3 import initialize,run_case

SOURCE=ROOT/'runs/pelican_e3_v1'
NAMES=['sindy_current','linear_history','sindy_history','motor_sindy']
HORIZONS=[.01,.05,.1,.25,.5,1.]


def load():
    frozen=json.loads((SOURCE/'FROZEN.json').read_text())
    for path,digest in frozen['source_sha256'].items():assert checksum(ROOT/path)==digest,path
    for path,digest in frozen['artifact_sha256'].items():assert checksum(SOURCE/path)==digest,path
    cfg=json.loads((SOURCE/'config.json').read_text())
    cfg.update(horizons_s=HORIZONS,max_speed_m_s=50.,max_body_rate_rad_s=100.)
    p=json.loads((SOURCE/'plant_parameters.json').read_text())
    motor=json.loads((SOURCE/'motor_model.json').read_text())
    models={m['name']:restore_model(m) for m in json.loads((SOURCE/'selected_models.json').read_text()) if m['name'] in NAMES}
    return cfg,p,motor,models


def log(out,event,**values):
    line=json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),event=event,**values))
    with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(line+'\n')
    print(line,flush=True)


def episode(folder,motor):
    with np.load(folder/'trajectory.npz') as f:t={key:f[key] for key in f.files}
    n=len(t['initial_history_commands'])
    x=np.concatenate([t['initial_history_states'],t['observation'][1:]])
    u=np.concatenate([t['initial_history_commands'],t['command']])
    u=np.vstack([u,u[-1]])
    seed=int(folder.name.split('_')[1])
    with np.load(SOURCE/'initial_conditions'/f'{seed}.npz') as f:
        prior=f['truth'][0,13:].copy(); expected=f['motor_prior'].copy()
    poles=np.asarray(motor['poles']);gain,bias=np.asarray(motor['coefficients']).T
    mhat=np.empty_like(u)
    for k,command in enumerate(u):
        prior=poles*prior+(1-poles)*(gain*command+bias);mhat[k]=prior
    np.testing.assert_allclose(mhat[n-1],expected,atol=1e-12)
    np.testing.assert_array_equal(x[n:],t['observation'])
    return {'x':x,'u':u,'motor_hat':mhat},t,n


def endpoint_errors(pred,true):
    dp=pred[:,:3]-true[:,:3];dv=pred[:,3:6]-true[:,3:6];dw=pred[:,10:13]-true[:,10:13]
    good=np.isfinite(pred).all(1)
    angle=np.full(len(pred),np.nan)
    angle[good]=(Rotation.from_quat(true[good,6:10]).inv()*Rotation.from_quat(pred[good,6:10])).magnitude()
    return {'position_sq_m2':np.sum(dp*dp,axis=1),'velocity_sq_m2_s2':np.sum(dv*dv,axis=1),
            'omega_sq_rad2_s2':np.sum(dw*dw,axis=1),'attitude_sq_rad2':angle**2,'failed':~good}


def static_correction(model,motor,cfg,p):
    hover=np.asarray(p['hover_command']);gain,bias=np.asarray(motor['coefficients']).T
    x=np.r_[0.,0.,1.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.]
    packed=pack_observation(np.tile(x,(21,1)),np.tile(hover,(20,1)),gain*hover+bias,model,cfg)
    fn,_=symbolic_transition(model,motor,cfg)
    predicted=np.asarray(fn(packed,hover)).ravel()
    residual=np.r_[(predicted[3:6]-x[3:6])/cfg['dt'],(predicted[10:13]-x[10:13])/cfg['dt']]
    corrected={**model,'y_mu':model['y_mu']-residual}
    fn2,_=symbolic_transition(corrected,motor,cfg)
    np.testing.assert_allclose(np.asarray(fn2(packed,hover)).ravel()[:13],x,atol=1e-11)
    return corrected,residual


def replay(out,cfg,p,motor,models):
    metrics=pd.read_csv(SOURCE/'tracking_per_run.csv');records=[];decomposition=[]
    pred_store=[];truth_store=[];index_store=[];start=time.monotonic()
    for row in metrics.itertuples(index=False):
        folder=SOURCE/'cases'/f'{row.trajectory}_{row.seed}_{row.controller}'
        e,t,n=episode(folder,motor)
        origins=np.arange(100,len(t['t'])-100,50)+n
        inputs=np.stack([e['u'][k:k+100] for k in origins])
        error=t['physical'][t['t']>=10,:3]-t['reference'][t['t']>=10,:3]
        mean=error.mean(0);bias2=float(mean@mean);variance=float(np.mean(np.sum((error-mean)**2,axis=1)))
        total=float(np.mean(np.sum(error**2,axis=1)))
        np.testing.assert_allclose(bias2+variance,total,atol=1e-14)
        decomposition.append(dict(trajectory=row.trajectory,seed=row.seed,controller=row.controller,
                                  late_rmse_m=np.sqrt(total),late_bias_norm_m=np.sqrt(bias2),
                                  late_demeaned_rmse_m=np.sqrt(variance),bias_fraction_mse=bias2/total,
                                  **{f'late_bias_{a}_m':mean[i] for i,a in enumerate('xyz')}))
        for name,model in models.items():
            x,h,prior=context_for_episode(e,origins,model,cfg)
            predictions,failed,_=rollout(x,h,inputs,prior,model,motor,cfg)
            for steps,pred in predictions.items():
                truth=e['x'][origins+steps];err=endpoint_errors(pred,truth)
                for j,k in enumerate(origins):
                    idx=dict(trajectory=row.trajectory,seed=row.seed,source_controller=row.controller,
                             model=name,origin_step=int(k-n),horizon_s=steps*cfg['dt'])
                    records.append({**idx,**{key:value[j] for key,value in err.items()}})
                    index_store.append(idx);pred_store.append(pred[j]);truth_store.append(truth[j])
        if (len(decomposition)%5)==0:log(out,'replay_progress',cases=len(decomposition),elapsed_s=time.monotonic()-start)
    frame=pd.DataFrame(records);frame.to_csv(out/'replay_per_window.csv',index=False)
    pd.DataFrame(index_store).to_csv(out/'replay_endpoint_index.csv',index=False)
    np.savez_compressed(out/'replay_endpoints.npz',predicted=pred_store,truth=truth_store)
    pd.DataFrame(decomposition).to_csv(out/'late_error_decomposition.csv',index=False)
    summaries=[]
    for columns in [['model','horizon_s'],['source_controller','model','horizon_s']]:
        for key,g in frame.groupby(columns):
            item=dict(zip(columns,key));item.setdefault('source_controller','all')
            item.update(windows=len(g),failures=int(g.failed.sum()))
            for field,label in [('position_sq_m2','position_rmse_m'),('velocity_sq_m2_s2','velocity_rmse_m_s'),
                                ('omega_sq_rad2_s2','omega_rmse_rad_s'),('attitude_sq_rad2','attitude_rmse_rad')]:
                percase=np.sqrt(g.groupby(['trajectory','seed','source_controller'])[field].mean())
                item[label]=percase.mean()
            summaries.append(item)
    pd.DataFrame(summaries).to_csv(out/'replay_summary.csv',index=False)


def observe_batch(previous,current,dt):
    x=current[:,:13].copy();x[:,3:6]=(current[:,:3]-previous[:,:3])/dt
    x[:,10:13]=(Rotation.from_quat(previous[:,6:10]).inv()*Rotation.from_quat(current[:,6:10])).as_rotvec()/dt
    return x


def sensitivities(out,cfg,p,motor,models):
    hover=np.asarray(p['hover_command']);gain,bias=np.asarray(motor['coefficients']).T
    state=np.r_[0.,0.,1.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,gain*hover+bias]
    e={'x':np.tile(state[:13],(200,1)),'u':np.tile(hover,(200,1)),
       'motor_hat':np.tile(gain*hover+bias,(200,1))}
    probes=[dict(label='hover',e=e,origin=100,physical=state,command=hover)]
    for kind in cfg['trajectories']:
        for source in ['pd']+NAMES:
            episode_data,t,n=episode(SOURCE/'cases'/f'{kind}_8301_{source}',motor)
            for second in [2,6,10,14,18]:
                k=round(second/cfg['dt'])
                probes.append(dict(label=f'{kind}_8301_{source}_t{second}',e=episode_data,origin=n+k,
                                   physical=t['physical'][k],command=t['command'][k-1]))
    branches=[];metadata=[]
    for i,probe in enumerate(probes):
        for amplitude in [.25,1.]:
            for rotor in range(4):
                for sign in [1,-1]:
                    u=probe['command'].copy();u[rotor]+=sign*amplitude
                    assert np.all(u>=p['command_lower']) and np.all(u<=p['command_upper'])
                    branches.append(u);metadata.append((i,amplitude,rotor,sign))
    commands=np.asarray(branches);physical=np.array([probes[i]['physical'] for i,_,_,_ in metadata])
    target={};horizons=[1,5,10,25,50]
    for k in range(1,51):
        new=step(physical,commands,cfg['dt'],p,cfg['integration_dt'])
        if k in horizons:target[k]=observe_batch(physical,new,cfg['dt'])
        physical=new
    predictions={}
    for name,model in models.items():
        contexts=[context_for_episode(probe['e'],np.array([probe['origin']]),model,cfg) for probe in probes]
        init=np.concatenate([contexts[i][0] for i,_,_,_ in metadata])
        hist=np.concatenate([contexts[i][1] for i,_,_,_ in metadata])
        prior=np.concatenate([contexts[i][2] for i,_,_,_ in metadata])
        predictions[name],failed,_=rollout(init,hist,np.repeat(commands[:,None,:],50,axis=1),prior,model,motor,cfg)
        log(out,'sensitivity_model_complete',model=name,branches=len(commands),failed=int(np.sum(failed<=50)))
    rows=[];arrays=[]
    for name in NAMES:
        for k in horizons:
            for j in range(0,len(metadata),2):
                i,amplitude,rotor,_=metadata[j]
                tr=(target[k][j]-target[k][j+1])/(2*amplitude)
                pr=(predictions[name][k][j]-predictions[name][k][j+1])/(2*amplitude)
                rows.append(dict(model=name,probe=probes[i]['label'],amplitude=amplitude,rotor=rotor+1,horizon_s=k*cfg['dt'],
                                 **{f'{v}_{a}':val[start+n] for v,val in [('plant_p',tr),('model_p',pr),
                                    ('plant_v',tr),('model_v',pr),('plant_w',tr),('model_w',pr)]
                                    for start in [0 if v.endswith('_p') else 3 if v.endswith('_v') else 10]
                                    for n,a in enumerate('xyz')}))
            arrays.append(dict(model=name,horizon_steps=k))
    frame=pd.DataFrame(rows);frame.to_csv(out/'input_sensitivity.csv',index=False)
    np.savez_compressed(out/'input_response_endpoints.npz',metadata=np.asarray(metadata),commands=commands,
                        **{f'plant_{k}':v for k,v in target.items()},
                        **{f'{name}_{k}':v for name,bytime in predictions.items() for k,v in bytime.items()})
    write_json(out/'input_probe_labels.json',[q['label'] for q in probes])
    summary=[]
    for region,f in [('hover',frame[frame.probe=='hover']),('trajectory',frame[frame.probe!='hover'])]:
        for (name,horizon,amp),g in f.groupby(['model','horizon_s','amplitude']):
            item=dict(region=region,model=name,horizon_s=horizon,amplitude=amp)
            for label in ['p','v','w']:
                true=g[[f'plant_{label}_{a}' for a in 'xyz']].to_numpy()
                pred=g[[f'model_{label}_{a}' for a in 'xyz']].to_numpy()
                mask=np.isfinite(pred).all(1)
                item[f'{label}_invalid_rows']=int((~mask).sum())
                true=true[mask];pred=pred[mask]
                item[f'{label}_relative_error']=np.linalg.norm(pred-true)/np.linalg.norm(true)
                item[f'{label}_gain_ratio']=np.linalg.norm(pred)/np.linalg.norm(true)
                item[f'{label}_cosine']=np.sum(pred*true)/(np.linalg.norm(pred)*np.linalg.norm(true))
            summary.append(item)
    pd.DataFrame(summary).to_csv(out/'input_sensitivity_summary.csv',index=False)
    residuals=[]
    for name,model in models.items():
        corrected,b=static_correction(model,motor,cfg,p)
        residuals.append(dict(model=name,**{f'bias_{i}':v for i,v in enumerate(b)}))
    pd.DataFrame(residuals).to_csv(out/'hover_corrections.csv',index=False)


def intervene(out,cfg,p,motor,models):
    target=out/'intervention';target.mkdir(exist_ok=False)
    cfg={**cfg,'duration_s':20.}
    write_json(target/'config.json',cfg)
    seeds=[9301,9302];kinds=['circle','figure8'];start=time.monotonic()
    initial={seed:initialize(seed,p,motor,cfg) for seed in seeds}
    (target/'initial_conditions').mkdir()
    for seed,value in initial.items():np.savez_compressed(target/'initial_conditions'/f'{seed}.npz',**value)
    controllers={};corrections={}
    for name,model in models.items():
        modified,residual=static_correction(model,motor,cfg,p);corrections[name]=residual
        controllers[(name,'original')]=MPC(model,motor,p,cfg)
        controllers[(name,'hover_corrected')]=MPC(modified,motor,p,cfg)
    write_json(target/'corrections.json',{key:value.tolist() for key,value in corrections.items()})
    write_json(target/'FROZEN.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=checksum(Path(__file__)),config_sha256=checksum(target/'config.json'),
        seeds=seeds,trajectories=kinds,variants=['original','hover_corrected'],exploratory=True))
    records=[]
    def logger(event,**values):log(out,event,**values)
    for kind in kinds:
        for seed in seeds:
            for name in NAMES:
                for variant in ['original','hover_corrected']:
                    log(out,'intervention_started',model=name,variant=variant,trajectory=kind,seed=seed)
                    m=run_case(name,kind,seed,controllers[(name,variant)],initial[seed],p,motor,cfg,
                               target/'cases'/f'{kind}_{seed}_{name}_{variant}',logger,start)
                    records.append({**m,'variant':variant});pd.DataFrame(records).to_csv(target/'per_run.csv',index=False)
                    log(out,'intervention_completed',variant=variant,**m)
    write_json(target/'COMPLETED.json',dict(cases=len(records),elapsed_s=time.monotonic()-start))


def main(args):
    cfg,p,motor,models=load();out=args.run_dir
    if args.stage=='analyze':
        out.mkdir(exist_ok=False)
        write_json(out/'STARTED.json',dict(utc=datetime.now(timezone.utc).isoformat(),
                   script_sha256=checksum(Path(__file__)),protocol_sha256=checksum(ROOT/'docs/PELICAN_E3_DIAGNOSTIC_PROTOCOL.md'),
                   source_frozen_sha256=checksum(SOURCE/'FROZEN.json'),evidence_type='POST-HOC SIMULATION DIAGNOSIS'))
        replay(out,cfg,p,motor,models);sensitivities(out,cfg,p,motor,models)
        write_json(out/'ANALYZED.json',dict(utc=datetime.now(timezone.utc).isoformat(),status='complete'))
    elif args.stage=='intervene':
        assert (out/'ANALYZED.json').exists()
        intervene(out,cfg,p,motor,models)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['analyze','intervene'],required=True)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/pelican_e3_diagnostic_v1')
    with threadpool_limits(limits=1):main(parser.parse_args())
