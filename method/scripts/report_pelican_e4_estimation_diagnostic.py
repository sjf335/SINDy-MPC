"""Paired descriptive diagnostics, fixed-path replay and truthful spatial figures."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from pelican_e2.data import state_features, advance
from pelican_e2.model import library
from run_pelican_e2 import restore_model, checksum, write_json
from verify_pelican_e4_estimation_diagnostic import independent_filter

NAMES={'pd':'PD','sindy_current':'Current SINDy','linear_history':'Linear history',
       'motor_sindy':'Motor + SINDy','sindy_history':'History SINDy'}
MODES=['nominal','raw','oracle_rates','causal_poly']
LABELS={'nominal':'No added noise','raw':'Raw noisy differences',
        'oracle_rates':'Oracle rates (diagnostic)','causal_poly':'Causal pose filter'}
COLORS={'nominal':'#666666','raw':'#D55E00','oracle_rates':'#009E73','causal_poly':'#0072B2'}
STYLES={'nominal':':','raw':'--','oracle_rates':'-.','causal_poly':'-'}


def load_npz(path):
    with np.load(path) as a:return {k:a[k] for k in a.files}


def pose_observations(truth,noise,dt):
    clean=truth[:,:13].copy()
    clean[1:,3:6]=np.diff(truth[:,:3],axis=0)/dt
    clean[1:,10:13]=(Rotation.from_quat(truth[:-1,6:10]).inv()*Rotation.from_quat(truth[1:,6:10])).as_rotvec()/dt
    raw=clean.copy();raw[:,:3]+=noise[:len(truth),:3]
    raw[:,6:10]=(Rotation.from_quat(truth[:,6:10])*Rotation.from_rotvec(noise[:len(truth),3:])).as_quat()
    raw[1:,3:6]=np.diff(raw[:,:3],axis=0)/dt
    raw[1:,10:13]=(Rotation.from_quat(raw[:-1,6:10]).inv()*Rotation.from_quat(raw[1:,6:10])).as_rotvec()/dt
    return clean,raw


def observation_metrics(x,clean,truth,t):
    result={}
    for suffix,mask in [('prefix',np.ones(len(t),bool)),('first_0p4s',t<=.4+1e-12)]:
        for label,cols in [('velocity',slice(3,6)),('rate',slice(10,13))]:
            result[label+'_observation_rmse_'+suffix]=float(np.sqrt(np.mean(np.sum((x[mask,cols]-clean[mask,cols])**2,axis=1))))
        result['position_observation_rmse_'+suffix]=float(np.sqrt(np.mean(np.sum((x[mask,:3]-truth[mask,:3])**2,axis=1))))
        result['attitude_observation_rmse_'+suffix]=float(np.sqrt(np.mean((Rotation.from_quat(truth[mask,6:10]).inv()*Rotation.from_quat(x[mask,6:10])).magnitude()**2)))
    return result


def fixed_path_replay(out,source,design,cfg,settings,motor,models):
    """Apply all observation modes to the SAME saved nominal state/command path.

    This prevents worse closed-loop motions from masquerading as estimator error.
    Additional channel substitutions are post-design exploratory sensitivity only.
    """
    rows=[];dt=cfg['dt'];window=design['filter_window_samples']
    scenario=next(s for s in settings['scenarios'] if s['name']=='pose_noise')
    gates=np.asarray(settings['observer']['residual_gate'])
    for seed in design['seeds']:
        init=load_npz(source/'initial_conditions'/f'{seed}.npz');n0=len(init['commands'])
        for kind in design['trajectories']:
            for name in design['controllers']:
                d=load_npz(source/'cases'/f'nominal_{kind}_{seed}_{name}'/'trajectory.npz')
                truth=np.vstack([init['truth'],d['physical'][1:]])
                noise=np.random.default_rng(settings['noise_seed_offset']+seed).standard_normal((len(truth),6))
                noise*=np.r_[np.full(3,scenario['position_noise_std_m']),np.full(3,scenario['rotation_noise_std_rad'])]
                clean,raw=pose_observations(truth,noise,dt)
                oracle=raw.copy();oracle[:,3:6]=clean[:,3:6];oracle[:,10:13]=clean[:,10:13]
                filtered=independent_filter(raw[:,:3],raw[:,6:10],dt,window)
                modes=dict(nominal=clean,raw=raw,oracle_rates=oracle,causal_poly=filtered)
                for label,columns in [('velocity_only',list(range(3,6))),('rate_only',list(range(10,13))),('pose_only',[0,1,2,6,7,8,9])]:
                    x=clean.copy();x[:,columns]=raw[:,columns];modes[label]=x
                commands=np.vstack([init['commands'],d['command']])
                target_clean=None
                for mode,x in modes.items():
                    r=dict(controller=name,trajectory=kind,seed=seed,mode=mode)
                    r.update(observation_metrics(x[n0:],clean[n0:],truth[n0:],d['t']))
                    if name!='pd':
                        model=models[name];inp=commands
                        if model['motor']:
                            poles=np.asarray(motor['poles']);gain,offset=np.asarray(motor['coefficients']).T
                            prior=np.asarray(init['motor_prior']);mhats=[]
                            # Only post-warmup indices are used by this no-history model.
                            inp=np.zeros_like(commands)
                            for k,u in enumerate(d['command']):
                                prior=poles*prior+(1-poles)*(gain*u+offset);inp[n0+k]=prior
                        z=(np.c_[state_features(x[:-1],cfg['gravity']),inp]-model['mu'])/model['scale']
                        idx=n0+np.arange(len(d['command']))
                        phi=library(z[idx],{lag:z[idx-lag] for lag in model['lags']},model['lags'],model['degree'])
                        target=(phi/model['feature_scale'])@model['coefficients'].T*model['y_scale']+model['y_mu']
                        if mode=='nominal':target_clean=target
                        delta=target-target_clean
                        r['predicted_force_perturbation_rms']=float(np.sqrt(np.mean(np.sum(delta[:,:3]**2,axis=1))))
                        r['predicted_angular_acceleration_perturbation_rms']=float(np.sqrt(np.mean(np.sum(delta[:,3:]**2,axis=1))))
                        prediction=advance(x[n0:-1],target,cfg)
                        residual=np.c_[Rotation.from_quat(x[n0:-1,6:10]).inv().apply((x[n0+1:,3:6]-prediction[:,3:6])/dt),
                                       (x[n0+1:,10:13]-prediction[:,10:13])/dt]
                        r['observer_rejection_fraction']=float(np.any(np.abs(residual)>gates,axis=1).mean())
                    rows.append(r)
    table=pd.DataFrame(rows);table.to_csv(out/'fixed_path_replay.csv',index=False)
    table.groupby(['mode','controller']).mean(numeric_only=True).drop(columns=['seed']).reset_index().to_csv(out/'fixed_path_replay_summary.csv',index=False)
    return table


def main(out):
    out=out.resolve();design=json.loads((out/'design.json').read_text());source=ROOT/design['base_run']
    cfg=json.loads((out/'e3_config.json').read_text());settings=json.loads((out/'config.json').read_text())
    motor=json.loads((out/'motor_model.json').read_text())
    models={k:restore_model(v) for k,v in json.loads((out/'corrected_models.json').read_text()).items()}
    new=pd.read_csv(out/'diagnostic_per_run.csv');new['origin']='new_diagnostic'
    new['case_path']=[(out/'cases'/v).relative_to(ROOT).as_posix() for v in new.case_id]
    baselines=pd.read_csv(source/'robustness_per_run.csv')
    baselines=baselines[baselines.scenario.isin(['nominal','pose_noise']) & baselines.seed.isin(design['seeds']) & baselines.trajectory.isin(design['trajectories'])].copy()
    assert len(baselines)==2*len(design['seeds'])*len(design['trajectories'])*len(design['controllers'])==40
    baselines['mode']=baselines.scenario.map({'nominal':'nominal','pose_noise':'raw'})
    baselines['bias_enabled']=True;baselines['origin']='frozen_E4'
    refs=[]
    for idx,r in baselines.iterrows():
        p=source/'cases'/f'{r.scenario}_{r.trajectory}_{r.seed}_{r.controller}'
        d=load_npz(p/'trajectory.npz');init=load_npz(source/'initial_conditions'/f'{r.seed}.npz')
        clean,_=pose_observations(np.vstack([init['truth'],d['physical'][1:]]),np.zeros((len(init['truth'])+len(d['command']),6)),cfg['dt'])
        for field,value in observation_metrics(d['observation'],clean[len(init['commands']):],d['physical'],d['t']).items():baselines.loc[idx,field]=value
        baselines.loc[idx,'case_path']=p.relative_to(ROOT).as_posix()
        baselines.loc[idx,'case_id']=p.name
        refs.append(dict(path=p.relative_to(ROOT).as_posix(),trajectory_sha256=checksum(p/'trajectory.npz'),metrics_sha256=checksum(p/'metrics.json')))
    combined=pd.concat([baselines,new],ignore_index=True)
    assert len(combined)==88
    combined.to_csv(out/'paired_all_runs.csv',index=False)
    write_json(out/'frozen_reference_cases.json',refs)
    summary=[]
    for (mode,on,name),g in combined.groupby(['mode','bias_enabled','controller']):
        periods=pd.concat([pd.read_csv(ROOT/p/'control.csv') for p in g.case_path])
        steps=np.round(g.duration_completed_s/cfg['dt']).sum()
        summary.append(dict(mode=mode,bias_enabled=bool(on),controller=name,cases=len(g),complete=int(g.complete.sum()),
            position_rmse_m=float(g.position_rmse_m.mean()) if g.complete.all() else None,
            max_position_error_m=float(g.max_position_error_m.max()),termination_time_min=float(g.duration_completed_s.min()),
            termination_time_max=float(g.duration_completed_s.max()),fallbacks=int(g.fallbacks.sum()),
            observer_rejection_fraction=float(g.observer_rejected.sum()/steps) if name!='pd' else None,
            full_period_p95_s=float(periods.full_control_period_s.quantile(.95)),
            over_period_fraction=float((periods.full_control_period_s>cfg['control_dt']).mean()),
            velocity_observation_rmse_first_0p4s=float(g.velocity_observation_rmse_first_0p4s.mean()),
            rate_observation_rmse_first_0p4s=float(g.rate_observation_rmse_first_0p4s.mean())))
    summary=pd.DataFrame(summary);summary.to_csv(out/'diagnostic_summary.csv',index=False)
    replay=fixed_path_replay(out,source,design,cfg,settings,motor,models)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.grid':True,'grid.alpha':.2})
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(figdir/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    # All four paired initial-state/trajectory paths are exported, including failures.
    for seed in design['seeds']:
        for kind in design['trajectories']:
            fig,axes=plt.subplots(2,5,figsize=(17,6.3),constrained_layout=True)
            for j,name in enumerate(design['controllers']):
                records=combined[(combined.seed==seed)&(combined.trajectory==kind)&(combined.controller==name)&combined.bias_enabled]
                reference=None
                for mode in MODES:
                    r=records[records['mode']==mode].iloc[0];d=load_npz(ROOT/r.case_path/'trajectory.npz')
                    if mode=='nominal':reference=d['reference'];ref_t=d['t']
                    axes[0,j].plot(d['physical'][:,0],d['physical'][:,1],color=COLORS[mode],ls=STYLES[mode],lw=1.25,label=LABELS[mode])
                    err=np.linalg.norm(d['physical'][:,:3]-d['reference'][:,:3],axis=1)*100
                    axes[1,j].plot(d['t'],err,color=COLORS[mode],ls=STYLES[mode],lw=1.15)
                    if not r.complete:
                        axes[0,j].plot(*d['physical'][-1,:2],marker='x',ms=7,color=COLORS[mode])
                        axes[1,j].plot(d['t'][-1],err[-1],marker='x',ms=7,color=COLORS[mode])
                axes[0,j].plot(reference[:,0],reference[:,1],'k--',lw=1,label='Reference')
                axes[0,j].set(title=NAMES[name],xlabel='x (m)',ylabel='y (m)');axes[0,j].set_aspect('equal',adjustable='datalim')
                axes[1,j].set(xlabel='Time (s)',ylabel='Position error (cm)',xlim=(0,20));axes[1,j].set_yscale('symlog',linthresh=1)
                axes[1,j].set_ylim(bottom=0)
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5)
            fig.suptitle(f'{kind}, paired seed {seed}; x = early termination; oracle uses simulation truth')
            save(fig,f'trajectory_{kind}_{seed}')
    # Dedicated 3-D tracking views, because XY alone can hide altitude failures.
    for kind in design['trajectories']:
        fig=plt.figure(figsize=(12,8),layout='constrained')
        for j,name in enumerate(design['controllers']):
            ax=fig.add_subplot(2,3,j+1,projection='3d')
            records=combined[(combined.seed==design['seeds'][0])&(combined.trajectory==kind)&(combined.controller==name)&combined.bias_enabled]
            for mode in MODES:
                r=records[records['mode']==mode].iloc[0];d=load_npz(ROOT/r.case_path/'trajectory.npz')
                ax.plot(*d['physical'][:,:3].T,color=COLORS[mode],ls=STYLES[mode],label=LABELS[mode],lw=1)
                if mode=='nominal':ax.plot(*d['reference'][:,:3].T,'k--',lw=1,label='Reference')
                if not r.complete:ax.scatter(*d['physical'][-1,:3],color=COLORS[mode],marker='x',s=25)
            ax.set(title=NAMES[name],xlabel='x (m)',ylabel='y (m)',zlabel='z (m)');ax.view_init(25,-55)
            ax.tick_params(labelsize=7,pad=1)
        h,l=fig.axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=5)
        fig.suptitle(f'{kind}, seed {design["seeds"][0]}; raw spatial paths, no smoothing')
        save(fig,'trajectory_3d_'+kind)
    # Completion and full-horizon RMSE, with no survivor-only averages.
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
    cells=[]
    for mode in MODES:
        cells.append([int(summary[(summary['mode']==mode)&summary.bias_enabled&(summary.controller==n)].iloc[0].complete) for n in design['controllers']])
    ax.imshow(cells,vmin=0,vmax=4,cmap='Blues',aspect='auto')
    for i,mode in enumerate(MODES):
        for j,name in enumerate(design['controllers']):
            r=summary[(summary['mode']==mode)&summary.bias_enabled&(summary.controller==name)].iloc[0]
            text=f'{int(r.complete)}/4 complete\n'+(f'{r.position_rmse_m*100:.2f} cm' if r.complete==4 else 'RMSE: incomplete')
            ax.text(j,i,text,ha='center',va='center',color='white' if r.complete==4 else 'black',fontsize=8)
    ax.set_xticks(range(5),[NAMES[n] for n in design['controllers']]);ax.set_yticks(range(4),[LABELS[m] for m in MODES]);ax.grid(False)
    ax.set_title('Paired observation diagnosis: 2 trajectories x 2 initial states (exploratory)')
    save(fig,'completion_and_rmse')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for ax,field,title in zip(axes,['velocity_observation_rmse_prefix','rate_observation_rmse_prefix'],['Velocity error (m/s)','Angular-rate error (rad/s)']):
        for i,mode in enumerate(['raw','causal_poly']):
            vals=[replay[(replay['mode']==mode)&(replay.controller==n)][field].mean() for n in design['controllers']]
            ax.bar(np.arange(5)+(i-.5)*.35,vals,width=.35,color=COLORS[mode],label=LABELS[mode])
        ax.set_xticks(range(5),[NAMES[n] for n in design['controllers']],rotation=25,ha='right');ax.set_ylabel(title);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Observation replay on identical saved nominal paths; vector RMS, full 20 s')
    save(fig,'fixed_path_observation_error')
    print(summary[['mode','bias_enabled','controller','cases','complete','position_rmse_m','observer_rejection_fraction']].to_string(index=False))
    print('Exported',len(list(figdir.glob('*.png'))),'PNG/PDF pairs')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    with threadpool_limits(limits=1):main(parser.parse_args().run_dir)
