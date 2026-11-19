"""Independently score paired time-domain traces for Section IV-C."""
from run_pelican_noise_pulse import SOURCE, DIAG, OUT, ROOT, SEEDS, KINDS, NAME, read, checksum, write_json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = ROOT.parent/'latex/figures'
FORCE = np.array([.4,-.25,.15])
AXIS = FORCE / np.linalg.norm(FORCE)


def archive(folder, seal):
    for name in ('trajectory.npz','metrics.json','control.csv'):
        path = folder/name
        expected = seal[path.relative_to(ROOT).as_posix()]['sha256'] if seal is not None else read(folder/'manifest.json')[name]
        assert checksum(path) == expected
    with np.load(folder/'trajectory.npz') as saved:
        data = {k:saved[k] for k in ('t','physical','observation','reference','world_acceleration','residual','bias_estimate','pose_noise','initial_history_states','initial_history_commands','accepted','clipped')}
    return data,read(folder/'metrics.json')


def error_metrics(d,meta,end):
    t=d['t']; np.testing.assert_allclose(t,.01*np.arange(len(t)),atol=1e-12)
    position=d['physical'][:,:3]-d['reference'][:,:3]
    error=np.linalg.norm(position,axis=1); signed=position@AXIS
    assert np.isclose(t[-1],20)==meta['complete']
    event=(t>=8)&(t<=end); after=(t>=end)&(t<=end+2)
    dwell=np.convolve((error>.03).astype(int),np.ones(101,dtype=int),'valid')
    indices=np.flatnonzero((t[:len(dwell)]>=end)&(dwell==0))
    recovery=t[indices[0]]-end if len(indices) else np.nan
    late=np.sqrt(np.mean(error[t>=end+2]**2))
    peak=error[event].max(); opposite=max(0.,-signed[after].min())
    if meta['complete']:
        np.testing.assert_allclose([peak,late],[meta['event_peak_error_m'],meta['post_event_rmse_m']],atol=1e-12)
    assert np.isfinite(recovery)==meta['recovered']
    if len(indices): np.testing.assert_allclose(recovery,meta['recovery_time_s'],atol=1e-12)
    return dict(event_peak_cm=100*peak,peak_at_s=t[event][np.argmax(error[event])],reverse_excursion_cm=100*opposite,
                recovery_s=recovery,late_rmse_cm=100*late,complete=meta['complete'],fallbacks=meta['fallbacks']),error


def projections(d):
    rotation=Rotation.from_quat(d['observation'][:,6:10]); estimate=rotation.apply(d['bias_estimate'][:,:3])@AXIS
    # Transition k -> k+1 uses the body axes at k. Its innovation becomes available at k+1.
    residual=np.r_[0.,rotation[:-1].apply(d['residual'][:,:3])@AXIS]
    return residual,estimate


def crossing(t, signal, mask):
    indices=np.flatnonzero(mask & signal)
    return t[indices[0]] if len(indices) else np.nan


def pulse_report(seal):
    rows=[]; lags=[]; traces=[]
    for kind in (*KINDS,'helix'):
        short = kind=='helix'; end=8.5 if short else 12.; arms=('short_off','short_on','short_fast') if short else ('pulse_off','pulse_on','pulse_fast')
        for seed in SEEDS:
            if not short:
                baseline,_=archive(SOURCE/'cases'/f'nominal_{kind}_{seed}_{NAME}',seal)
            anchor=None
            for arm in arms:
                reused=arm=='pulse_on'; folder=SOURCE/'cases'/f'force_pulse_{kind}_{seed}_{NAME}' if reused else OUT/'cases'/f'{arm}_{kind}_{seed}'
                d,meta=archive(folder,seal if reused else None); t=d['t']
                if anchor is None:
                    anchor=d
                for key in ('initial_history_states','initial_history_commands','reference'):
                    np.testing.assert_array_equal(d[key],anchor[key])
                np.testing.assert_array_equal(d['physical'][0],anchor['physical'][0]); assert not np.any(d['pose_noise'])
                force=np.zeros_like(d['world_acceleration']);force[(t[:-1]>=8)&(t[:-1]<end)]=FORCE
                np.testing.assert_array_equal(d['world_acceleration'],force)
                control=pd.read_csv(folder/'control.csv'); used=control[[f'bias{i}' for i in range(6)]].to_numpy()
                expected=np.zeros_like(used) if arm.endswith('off') else d['bias_estimate'][control.step]
                np.testing.assert_allclose(used,expected,atol=1e-12)
                result,error=error_metrics(d,meta,end)
                rows.append(dict(arm=arm,trajectory=kind,seed=seed,**result))
                if short: continue
                if reused:
                    np.testing.assert_array_equal(d['physical'][:801],baseline['physical'][:801]); base_r,base_b=projections(baseline)
                    r,b=projections(d); r-=base_r; b-=base_b
                    plateau=b[(t>=11)&(t<12)].mean(); half=plateau*.5; target=.5*np.linalg.norm(FORCE)
                    lags.append(dict(trajectory=kind,seed=seed,plateau_m_s2=plateau,
                        half_injected_crossing_s=crossing(t,b>=target,(t>=8)&(t<12))-8,
                        half_plateau_rise_s=crossing(t,b>=half,(t>=8)&(t<12))-8,
                        half_plateau_decay_s=crossing(t,b<=half,t>=12)-12,
                        first_residual_m_s2=r[801],first_estimate_m_s2=b[801]))
                else: r=np.full(len(t),np.nan);b=r
                traces.append(pd.DataFrame(dict(arm=arm,trajectory=kind,seed=seed,t=t,position_error_cm=100*error,
                                               paired_residual_m_s2=r,paired_estimate_m_s2=b)))
    pairs=pd.DataFrame(rows);pairs.to_csv(OUT/'pulse_per_case.csv',index=False)
    summary=pairs.groupby('arm').agg(cases=('complete','size'),completed=('complete','sum'),recovered=('recovery_s','count'),
        event_peak_cm=('event_peak_cm','mean'),reverse_excursion_cm=('reverse_excursion_cm','mean'),recovery_s=('recovery_s','mean'),
        late_rmse_cm=('late_rmse_cm','mean'),fallbacks=('fallbacks','sum'))
    summary.loc[summary.recovered!=summary.cases,'recovery_s']=np.nan
    summary.to_csv(OUT/'pulse_summary.csv'); lag=pd.DataFrame(lags);lag.to_csv(OUT/'observer_lag_per_case.csv',index=False)
    mean_traces=pd.concat(traces).groupby(['arm','t'],as_index=False)[['position_error_cm','paired_residual_m_s2','paired_estimate_m_s2']].mean()
    mean_traces.to_csv(OUT/'pulse_mean_traces.csv',index=False)
    return summary,lag,mean_traces


def noise_report(seal):
    rows=[]
    for kind in KINDS:
        for seed in SEEDS:
            template,_=archive(SOURCE/'cases'/f'pose_noise_{kind}_{seed}_{NAME}',seal)
            for arm,scale in [('nominal',0.),('noise_low',.1),('noise_mid',.5),('pose_noise',1.)]:
                reused=arm in ('nominal','pose_noise')
                for ctrl in (read(SOURCE/'config.json')['controllers'] if reused else [NAME]):
                    folder=SOURCE/'cases'/f'{arm}_{kind}_{seed}_{ctrl}' if reused else OUT/'cases'/f'{arm}_{kind}_{seed}'
                    d,m=archive(folder,seal if reused else None)
                    np.testing.assert_allclose(d['pose_noise'],scale*template['pose_noise'],atol=1e-15,rtol=1e-12)
                    np.testing.assert_array_equal(d['physical'][0],template['physical'][0]);assert not np.any(d['world_acceleration'])
                    error=np.linalg.norm(d['physical'][:,:3]-d['reference'][:,:3],axis=1)
                    complete=len(error)==2001 and m['termination_reason']=='completed'; assert complete==m['complete']
                    if complete: np.testing.assert_allclose(np.sqrt(np.mean(error**2)),m['position_rmse_m'],atol=1e-12)
                    else: assert m['position_rmse_m'] is None
                    rows.append(dict(noise_mm_mrad=scale,controller=ctrl,trajectory=kind,seed=seed,complete=complete,duration_s=d['t'][-1],
                        position_rmse_cm=100*m['position_rmse_m'] if complete else np.nan,termination=m['termination_reason'],fallbacks=m['fallbacks']))
    data=pd.DataFrame(rows); data.to_csv(OUT/'noise_per_case.csv',index=False)
    summary=data.groupby(['controller','noise_mm_mrad']).agg(cases=('complete','size'),completed=('complete','sum'),
        min_duration_s=('duration_s','min'),median_duration_s=('duration_s','median'),max_duration_s=('duration_s','max'),
        rmse_cm=('position_rmse_cm','mean'),fallbacks=('fallbacks','sum'))
    summary.loc[summary.cases!=summary.completed,'rmse_cm']=np.nan; summary.to_csv(OUT/'noise_summary.csv')
    diag_seal=read(DIAG/'report_verification.json'); assert diag_seal['status']=='PASS'
    file=DIAG/'paired_all_runs.csv'; assert checksum(file)==diag_seal['inputs_sha256'][file.name]
    frame=pd.read_csv(file); diag=frame[frame.bias_enabled & (frame.controller==NAME)].copy()
    assert set(diag.seed)==set(SEEDS[:2]) and len(diag)==16
    results=diag.groupby('mode').agg(cases=('complete','size'),completed=('complete','sum'),rmse_m=('position_rmse_m','mean'))
    results.loc[results.cases!=results.completed,'rmse_m']=np.nan;results.to_csv(OUT/'estimation_diagnostic.csv')
    return summary,results


def plot(traces):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(251.65 / 72, 3.20), sharex=True)
    fig.subplots_adjust(left=.225, right=.985, top=.98, bottom=.265, hspace=.18)
    full=traces[traces.arm=='pulse_on']; t=full.t.to_numpy()
    top.plot(t,np.where((t>=8)&(t<12),np.linalg.norm(FORCE),0),c='.35',ls=':',label='Applied pulse')
    top.plot(t,full.paired_residual_m_s2,c='#C66914',lw=.9,ls='--',label='Residual')
    top.plot(t,full.paired_estimate_m_s2,c='#126591',lw=1.4,label='Estimate')
    top.set_ylabel('Acceleration\n(m/s²)')
    bottom_limit = min(-.15, full.paired_residual_m_s2.min() - .025)
    top.set_ylim(bottom_limit, .68)
    top.set_yticks([-.25, 0, .25, .5] if bottom_limit <= -.25 else [0, .25, .5])
    responses = [
        ('pulse_off', 'Off', '#333333', (0, (7, 2)), None),
        ('pulse_on', '0.50 s', '#00856A', '-.', None),
        ('pulse_fast', '0.25 s', '#9C438C', '-', 'o'),
    ]
    for arm, label, color, style, marker in responses:
        part = traces[traces.arm == arm]
        bottom.plot(part.t, part.position_error_cm, label=label, color=color,
                    linestyle=style, linewidth=1.25, marker=marker, markevery=75,
                    markersize=2.8, markerfacecolor='white', markeredgewidth=.75)
    bottom.set_ylim(0, 19.5)
    bottom.set_ylabel('Position error\n(cm)')
    bottom.set_xlabel('Time (s)')
    # Matplotlib fills columns first. Keep each panel's entries on one row.
    upper_lines = top.get_legend_handles_labels()[0]
    lower_lines = bottom.get_legend_handles_labels()[0]
    handles = [line for column in zip(upper_lines, lower_lines) for line in column]
    legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .015),
                        ncols=3, fontsize=8, frameon=True, fancybox=False,
                        edgecolor='black', facecolor='white', framealpha=1,
                        borderpad=.5, handlelength=2.4, columnspacing=1.2, labelspacing=.5)
    legend.get_frame().set_linewidth(.6)
    for ax in (top,bottom):
        ax.axvspan(8,12,color='.5',alpha=.08,zorder=-2);ax.set_xlim(7.5,15.5);ax.grid(axis='y',alpha=.24)
    fig.savefig(FIG/'noise_pulse_lag.pdf',metadata={'CreationDate':None});fig.savefig(OUT/'noise_pulse_lag.png',dpi=260);plt.close(fig)


if __name__=='__main__':
    seal=read(SOURCE/'artifact_hashes.json')['files']
    noise,diagnostics=noise_report(seal);pulse,lag,traces=pulse_report(seal);plot(traces)
    assert pulse.cases.sum()==45 and len(lag)==10 and len(list((OUT/'cases').glob('*/manifest.json')))==55
    write_json(OUT/'verification.json',dict(status='PASS',new_cases=55,noise_case_records=int(noise.cases.sum()),pulse_case_records=45,
        matched_lag_pairs=10,diagnostic_case_records=int(diagnostics.cases.sum()),paired_noise_and_initial_conditions=True,
        pulse_inputs_and_applied_bias_verified=True,source_summary_sha256=checksum(OUT/'pulse_summary.csv')))
    print(noise.to_string());print(diagnostics.to_string());print(pulse.to_string());print(lag.mean(numeric_only=True).to_string())
