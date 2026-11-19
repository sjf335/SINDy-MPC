"""Create descriptive flight-paired statistics and figures from frozen E2 results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits
from pelican_e2.model import context_for_episode, rollout
from run_pelican_e2 import checksum, restore_model, write_json
from verify_pelican_e2 import load_episode, verify_freeze

ORDER = ['sindy_current','linear_history','sindy_history','motor_sindy','constant_twist']
LABELS = {'sindy_current':'Current SINDy','linear_history':'Linear history',
          'sindy_history':'History SINDy','motor_sindy':'Motor + SINDy','constant_twist':'Constant twist'}
COLORS = {'sindy_current':'#D55E00','linear_history':'#0072B2','sindy_history':'#009E73',
          'motor_sindy':'#CC79A7','constant_twist':'#666666'}
STYLES = {'sindy_current':'--','linear_history':'-.','sindy_history':'-',
          'motor_sindy':(0,(5,1,1,1)),'constant_twist':':'}
MARKERS = dict(zip(ORDER,['s','^','o','D','x']))
METRICS = ['velocity','omega','position','attitude']
UNITS = ['m/s','rad/s','m','rad']


def save_figure(fig, folder, name):
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(folder/(name+'.png'),dpi=300,bbox_inches='tight')
    with Image.open(folder/(name+'.png')) as im:
        im.convert('L').save(folder/(name+'_grayscale.png'),dpi=(300,300))
    plt.close(fig)


def main(args):
    run = args.run_dir
    freeze,splits = verify_freeze(run)
    verification = json.loads((run/'verification.json').read_text())
    assert verification['status']=='PASS'
    config = json.loads((run/'config.json').read_text())
    motor = json.loads((run/'motor_model.json').read_text())
    models = {m['name']:restore_model(m) for m in json.loads((run/'models/selected_models.json').read_text())}
    perflight = pd.read_csv(run/'test/prediction_per_flight.csv')
    windows = pd.read_csv(run/'test/prediction_per_window.csv')
    summary = pd.read_csv(run/'test/prediction_summary.csv')
    out,figures = run/'analysis',run/'figures'
    out.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    ids = sorted(splits['test'])
    rng = np.random.default_rng(config['bootstrap_seed'])
    draws = rng.integers(0,len(ids),size=(config['bootstrap_repetitions'],len(ids)))
    np.savez_compressed(out/'bootstrap_draws.npz',flight_ids=ids,draws=draws)
    ci_rows = []
    for model in ORDER:
        for horizon in config['horizons_s']:
            subset = perflight[(perflight.model==model)&(perflight.horizon_s==horizon)].set_index('flight_id').loc[ids]
            for metric in METRICS:
                values = subset[metric+'_rmse'].to_numpy()
                low,high = np.quantile(values[draws].mean(1),[.025,.975])
                ci_rows.append({'model':model,'horizon_s':horizon,'metric':metric,
                                'mean':values.mean(),'lower_95':low,'upper_95':high,
                                'failed_windows':int(subset.failed_windows.sum()),
                                'conditional_on_success':bool(subset.failed_windows.sum())})
    intervals = pd.DataFrame(ci_rows)
    intervals.to_csv(out/'metric_intervals.csv',index=False)
    primary = perflight[perflight.horizon_s==config['primary_horizon_s']]
    comparisons = []
    for baseline in [m for m in ORDER if m!='sindy_history']:
        b = primary[primary.model==baseline].set_index('flight_id').loc[ids]
        h = primary[primary.model=='sindy_history'].set_index('flight_id').loc[ids]
        assert b.failed_windows.sum()==h.failed_windows.sum()==0
        for metric in METRICS:
            base,target = b[metric+'_rmse'].to_numpy(),h[metric+'_rmse'].to_numpy()
            delta = base-target
            delta_draws = delta[draws].mean(1)
            relative_draws = 100*delta_draws/base[draws].mean(1)
            low,high = np.quantile(delta_draws,[.025,.975])
            rlow,rhigh = np.quantile(relative_draws,[.025,.975])
            comparisons.append({'baseline':baseline,'candidate':'sindy_history','metric':metric,
                                'horizon_s':config['primary_horizon_s'],'flights':len(ids),
                                'baseline_mean':base.mean(),'candidate_mean':target.mean(),
                                'improvement_absolute':delta.mean(),'lower_95':low,'upper_95':high,
                                'improvement_percent':100*delta.mean()/base.mean(),
                                'percent_lower_95':rlow,'percent_upper_95':rhigh,
                                'flights_improved':int((delta>0).sum())})
    pairs = pd.DataFrame(comparisons)
    pairs.to_csv(out/'primary_paired_bootstrap.csv',index=False)
    main_table = summary[(summary.horizon_s==config['primary_horizon_s'])&summary.model.isin(ORDER)].copy()
    main_table['regression_nonzero_coefficients'] = main_table.model.map(lambda name:models[name]['nonzero_coefficients'])
    main_table['history_span_ms'] = main_table.model.map(lambda name:1000*config['dt']*max(models[name]['lags'],default=0))
    main_table.to_csv(out/'primary_results.csv',index=False)

    # Reconstructable coefficients with feature order, output scaling and bias.
    coef_rows = []
    for name in ORDER:
        m = models[name]
        if m['family']=='constant_twist': continue
        feature_names = ['1']+[f'z{i}' for i in range(13)]
        if m['degree']==2:
            feature_names += [f'z{i}*z{j}' for i in range(13) for j in range(i,13)]
        for lag in m['lags']:
            feature_names += [f'z{i}(k-{lag})-z{i}(k)' for i in [0,1,2,3,4,5,9,10,11,12]]
            if m['degree']==2:
                feature_names += [f'z{i}(k-{lag})^2-z{i}(k)^2' for i in range(9,13)]
        for axis in range(6):
            for j,feature in enumerate(feature_names):
                coef_rows.append({'model':name,'target':['f_body_x','f_body_y','f_body_z','angular_acc_x','angular_acc_y','angular_acc_z'][axis],
                                 'feature_index':j,'feature':feature,
                                 'regression_coefficient':m['coefficients'][axis,j],
                                 'output_unit_coefficient':m['y_scale'][axis]*m['coefficients'][axis,j]/m['feature_scale'][j],
                                 'output_mean_added_once':m['y_mu'][axis]})
    pd.DataFrame(coef_rows).to_csv(out/'coefficients_long.csv',index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                         'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,
                         'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.5})
    fig,axes = plt.subplots(2,2,figsize=(7.2,5.6))
    for ax,metric,unit,letter in zip(axes.flat,METRICS,UNITS,'abcd'):
        for name in ORDER:
            s = intervals[(intervals.model==name)&(intervals.metric==metric)].sort_values('horizon_s')
            ax.plot(s.horizon_s,s['mean'],color=COLORS[name],linestyle=STYLES[name],
                    marker=MARKERS[name],markersize=4,label=LABELS[name])
            ax.fill_between(s.horizon_s,s.lower_95,s.upper_95,color=COLORS[name],alpha=.09,linewidth=0)
        ax.set(xscale='log',yscale='log',xlabel='Prediction horizon (s)',
               ylabel=f'Flight-mean endpoint RMSE ({unit})',title=f'({letter}) {metric.capitalize()}')
        ax.set_xticks(config['horizons_s'],labels=['0.01','0.1','0.25','0.5','1.0'])
        ax.grid(alpha=.15,which='major')
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.00))
    fig.suptitle('REAL-DATA PREDICTION | 11 held-out flights',y=1.045,fontsize=11)
    fig.text(.5,-.02,'Bands: descriptive 95% flight bootstrap. At 1 s, Current / Motor SINDy exclude 2 / 48 failed windows.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,0,1,.92))
    save_figure(fig,figures,'prediction_horizons')

    fig,axes = plt.subplots(1,2,figsize=(7.2,4.2),sharey=True)
    primary_velocity = primary.pivot(index='flight_id',columns='model',values='velocity_rmse').loc[ids]
    primary_velocity.to_csv(out/'primary_velocity_per_flight.csv')
    for ax,base,letter in zip(axes,['sindy_current','linear_history'],'ab'):
        a,b = primary_velocity[base],primary_velocity.sindy_history
        y = np.arange(len(ids))
        ax.hlines(y,np.minimum(a,b),np.maximum(a,b),color='#aaaaaa',lw=1.5)
        ax.scatter(a,y,color=COLORS[base],marker=MARKERS[base],s=24,zorder=3,label=LABELS[base])
        ax.scatter(b,y,color=COLORS['sindy_history'],marker='o',s=24,zorder=4,label='History SINDy')
        ax.set(xlim=(0,1.85),ylim=(-.7,len(ids)+1.3),xlabel='Velocity endpoint RMSE (m/s)',
               title=f'({letter}) {int((b<a).sum())}/11 flights improved')
        ax.set_yticks(y,labels=ids); ax.grid(alpha=.15,axis='x')
        ax.legend(loc='upper left',frameon=False,fontsize=8)
    axes[0].set_ylabel('Test flight ID')
    fig.suptitle('REAL-DATA PREDICTION | paired 0.5 s velocity errors',fontsize=10)
    fig.tight_layout()
    save_figure(fig,figures,'primary_per_flight')

    fig,ax = plt.subplots(figsize=(7.2,3.2))
    for name in ORDER:
        s = summary[summary.model==name].sort_values('horizon_s')
        ax.plot(s.horizon_s,100*s.mean_flight_failure_rate,color=COLORS[name],
                ls=STYLES[name],marker=MARKERS[name],label=LABELS[name])
    ax.set(xlabel='Prediction horizon (s)',ylabel='Mean flight failure rate (%)',
           title='REAL-DATA PREDICTION | divergence retained as failure')
    ax.set_xticks(config['horizons_s'],labels=['0.01','0.1','0.25','0.5','1.0'])
    ax.set_ylim(-.1,2.7); ax.grid(alpha=.15)
    ax.annotate('Motor + SINDy: 48 / 2515 windows',(1,2.1367276),xytext=(-190,12),textcoords='offset points',fontsize=8)
    ax.annotate('Current SINDy: 2 / 2515 windows',(1,.08912656),xytext=(-190,20),textcoords='offset points',fontsize=8)
    ax.legend(loc='upper left',ncol=2,frameon=False)
    fig.tight_layout(); save_figure(fig,figures,'prediction_failures')

    baseline = primary[primary.model=='sindy_current'].sort_values(['velocity_rmse','flight_id'])
    examples = []
    for designation,fid in [('median',int(baseline.iloc[len(baseline)//2].flight_id)),
                            ('worst',int(baseline.iloc[-1].flight_id))]:
        sample = windows[(windows.model=='sindy_current')&(windows.flight_id==fid)&
                         (windows.horizon_s==config['primary_horizon_s'])].sort_values(['velocity_error','origin_index'])
        chosen = sample.iloc[len(sample)//2]
        origin = int(chosen.origin_index)
        episode = load_episode(run,fid)
        steps = round(max(config['horizons_s'])/config['dt'])
        truth = episode['x'][origin:origin+steps+1]
        paths,failures = {},{}
        for name in ORDER:
            m = models[name]
            x,history,prior = context_for_episode(episode,np.array([origin]),m,config)
            _,failed,path = rollout(x,history,episode['u'][origin:origin+steps][None],prior,m,motor,config,keep_all=True)
            paths[name] = path[0]
            if failed[0] <= steps: failures[name] = float(failed[0]*config['dt'])
        time = np.arange(steps+1)*config['dt']
        records = []
        for name,path in [('observed',truth),*paths.items()]:
            for k,t in enumerate(time):
                records.append({'flight_id':fid,'origin_index':origin,'model':name,'time_from_origin_s':t,
                                **dict(zip(['vx','vy','vz','wx','wy','wz'],path[k,[3,4,5,10,11,12]]))})
        pd.DataFrame(records).to_csv(out/f'example_{designation}_flight_{fid:02d}.csv',index=False)
        np.savez_compressed(out/f'example_{designation}_flight_{fid:02d}.npz',time=time,truth=truth,**paths)
        fig,axes = plt.subplots(2,3,figsize=(7.2,4.8),sharex=True)
        for ax,col,label,unit,letter in zip(axes.flat,[3,4,5,10,11,12],['v_x','v_y','v_z',r'\omega_x',r'\omega_y',r'\omega_z'],
                                           ['m/s']*3+['rad/s']*3,'abcdef'):
            ax.plot(time,truth[:,col],color='black',lw=1.8,label='Observed')
            for name in ORDER:
                ax.plot(time,paths[name][:,col],color=COLORS[name],ls=STYLES[name],lw=1.2,label=LABELS[name])
            ax.set(ylabel=f'${label}$ ({unit})',title=f'({letter})'); ax.grid(alpha=.15)
        for ax in axes[-1]: ax.set_xlabel('Prediction time (s)')
        handles,labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.00))
        fig.suptitle(f'REAL-DATA PREDICTION | {designation} baseline-error flight {fid} | start {episode["t"][origin]:.2f} s',
                     y=1.045,fontsize=10)
        if failures:
            note = '; '.join(f'{LABELS[name]} at {t:.2f} s' for name,t in failures.items())
            fig.text(.5,-.025,'Divergence termination: '+note+'. Subsequent predictions are missing.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,0,1,.9)); save_figure(fig,figures,f'example_{designation}')
        examples.append({'designation':designation,'flight_id':fid,'origin_index':origin,
                         'raw_origin_index':int(episode['raw_index'][origin]),'origin_t_s':episode['t'][origin],
                         'baseline_0p5s_window_velocity_error':float(chosen.velocity_error),
                         'divergence_time_s':failures})
    write_json(out/'representative_selection.json',{'rule':freeze['predeclared_plot_selection'],'examples':examples})
    result = {'status':'completed','generated_utc':datetime.now(timezone.utc).isoformat(),
              'evidence_type':'real-flight log conditional prediction; no new closed-loop control',
              'primary_horizon_s':config['primary_horizon_s'],'test_flights':ids,
              'common_origins':int(main_table.iloc[0].windows),'training_transitions':freeze['training_transitions'],
              'primary_results':main_table.to_dict(orient='records'),
              'primary_paired_bootstrap':comparisons,'bootstrap_unit':'whole flight, paired across methods',
              'bootstrap_repetitions':config['bootstrap_repetitions'],'bootstrap_seed':config['bootstrap_seed'],
              'uncertainty_scope':'Descriptive pointwise intervals; collection-session IDs unknown; no p-values or multiplicity-adjusted claims.',
              'linear_matched_equals_independent_selection':True,'representative_examples':examples,
              'verification':{k:v for k,v in verification.items() if k!='reference_details'},
              'limitations':['Upstream smoothed poses; causal preparation does not undo original filtering.',
                             'Future recorded commands condition the forecast; no future measured states or motor speed used.',
                             'One train/validation/test split; no across-training-subset uncertainty.',
                             'No GRU/RNN comparison or state-only versus command-only history ablation in E2.',
                             'Motor commands and reported speed are vendor codes; fitted coefficients are not identified physical parameters.',
                             'No MPC tracking, closed-loop robustness, or real-time deployment claim.']}
    write_json(out/'report.json',result)
    print(main_table[['model','velocity_rmse_mean','omega_rmse_mean','position_rmse_mean','attitude_rmse_mean']].to_string(index=False))
    print(pairs[pairs.metric=='velocity'][['baseline','improvement_percent','lower_95','upper_95','flights_improved']].to_string(index=False))
    print(json.dumps({'examples':examples,'figures':len(list(figures.glob('*.pdf')))},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/pelican_e2_v1')
    with threadpool_limits(limits=1): main(parser.parse_args())
