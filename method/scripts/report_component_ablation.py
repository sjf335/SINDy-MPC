"""Recompute the ablation outcomes from per-step signals before exporting tables."""
from pathlib import Path
import numpy as np
import pandas as pd
from run_component_ablation import ROOT, E2, OUT, TRACK, GROUPS, KINDS, SEEDS, read, checksum, write_json
from history_feature_control import CONTROL

LABEL = {'full':'Full model', 'no_history':'Without history', 'no_calibration':'Without hover calibration',
         'no_bias':'Without online bias', 'none':'Without all three', 'shift':'Shifted-history control'}


def collect_tracking():
    frozen = pd.read_csv(TRACK/'validate/tracking_per_run.csv')
    full_ids = frozen[frozen.controller=='sindy_history'].set_index(['trajectory','seed']).case_id.to_dict()
    sealed = read(TRACK/'artifact_hashes.json')['files']
    collected=[]
    for kind in KINDS:
        for seed in SEEDS:
            original = TRACK/'validate/cases'/full_ids[kind,seed]
            with np.load(original/'trajectory.npz') as d:
                reference = d['reference']; initial_state=d['physical'][0]
                initial_history = d['initial_history_states']; initial_u=d['initial_history_commands']
            for group in LABEL:
                if group=='full':
                    folder=original
                    for filename in ['control.csv','trajectory.npz','metrics.json']:
                        path=folder/filename
                        assert checksum(path)==sealed[path.relative_to(ROOT).as_posix()]['sha256']
                else:
                    folder = CONTROL/'cases'/f'{kind}_{seed}' if group=='shift' else OUT/'cases'/f'{group}_{kind}_{seed}'
                    for name,digest in read(folder/'manifest.json').items():
                        assert checksum(folder/name)==digest
                m=read(folder/'metrics.json'); c=pd.read_csv(folder/'control.csv')
                with np.load(folder/'trajectory.npz') as d:
                    t=d['t']; error_vec=d['physical'][:,:3]-d['reference'][:,:3]; position_norm=np.linalg.norm(error_vec,axis=1)
                    np.testing.assert_array_equal(d['physical'][0],initial_state)
                    np.testing.assert_array_equal(d['initial_history_states'],initial_history)
                    np.testing.assert_array_equal(d['initial_history_commands'],initial_u)
                    np.testing.assert_array_equal(d['reference'],reference[:len(t)])
                    np.testing.assert_array_equal(t,np.arange(len(t))*.01)
                    step_indices=c.step.to_numpy(dtype=int)
                    np.testing.assert_array_equal(c[[f'bias{i}' for i in range(6)]].to_numpy(),np.zeros((len(c),6))) if group in ('none','no_bias') else np.testing.assert_allclose(c[[f'bias{i}' for i in range(6)]].to_numpy(),d['bias_estimate'][step_indices],atol=1e-12)
                    np.testing.assert_allclose(c[[f'u{i}' for i in range(1,5)]].to_numpy(),d['command'][step_indices],atol=1e-12)
                assert m['trajectory']==kind and m['seed']==seed
                complete = len(t)==2001 and m['termination_reason']=='completed'; assert m['complete']==complete
                metrics = dict(rmse_cm=np.nan,late_rmse_cm=np.nan,late_bias_cm=np.nan)
                if complete:
                    for key, field, value in [('rmse_cm','position_rmse_m',np.sqrt(np.mean(position_norm**2))),
                        ('late_rmse_cm','late_rmse_m',np.sqrt(np.mean(position_norm[t>=10]**2))),
                        ('late_bias_cm','late_bias_norm_m',np.linalg.norm(error_vec[t>=10].mean(axis=0)))]:
                        np.testing.assert_allclose(value,m[field],rtol=1e-11,atol=1e-12);metrics[key]=100*value
                collected.append(dict(group=group,trajectory=kind,seed=seed,complete=complete,fallbacks=m['fallbacks'],
                    origin='frozen' if group=='full' else 'new',case_path=folder.relative_to(ROOT).as_posix(),**metrics))
    return pd.DataFrame(collected)


def main():
    cases=collect_tracking(); cases.to_csv(OUT/'tracking_per_case.csv',index=False)
    summary=[]
    for group in LABEL:
        frames=cases[cases.group==group]
        for kind in (*KINDS,'all'):
            f=frames if kind=='all' else frames[frames.trajectory==kind]
            metrics={col:f[col].mean() if f.complete.all() else np.nan for col in ['rmse_cm','late_rmse_cm','late_bias_cm']}
            summary.append(dict(group=group,trajectory=kind,cases=len(f),completed=f.complete.sum(),fallbacks=f.fallbacks.sum(),**metrics))
    tracking=pd.DataFrame(summary);tracking.to_csv(OUT/'tracking_summary.csv',index=False)
    prediction=pd.read_csv(OUT/'prediction_summary.csv').query('horizon_s == 0.5').set_index('group')
    shift=pd.read_csv(CONTROL/'prediction_per_flight.csv'); assert len(shift)==11 and shift.windows.sum()==2515
    prediction.loc['shift',['position_m','velocity_m_s','windows','failed']]=[shift.position_m.mean(),shift.velocity_m_s.mean(),shift.windows.sum(),shift.failed.sum()]
    result=tracking.query("trajectory=='all'").set_index('group').join(prediction[['velocity_m_s','position_m','windows','failed']]);result=result.loc[list(LABEL)]
    result.to_csv(OUT/'ablation_results.csv')
    lines=[]
    for g,r in result.iterrows():
        if g=='shift':lines.append(r'\midrule')
        lines.append(f"{LABEL[g]} & {r.velocity_m_s:.3f} & {100*r.position_m:.2f} & {r.rmse_cm:.2f} " + r'\\')
    table = r'''% Generated by method/scripts/report_component_ablation.py.
\begin{table}[!ht]
  \centering
  \caption{Component Ablations on Flight Prediction and Simulated Tracking}
  \label{tab:component-ablation}
  \footnotesize
  \setlength{\tabcolsep}{3.5pt}
  \begin{tabular}{@{}lrrr@{}}
    \toprule
    & \multicolumn{2}{c}{Prediction at 0.5 s} & Tracking \\
    \cmidrule(lr){2-3}
    Variant & Vel. (m/s) & Pos. (cm) & Pos. (cm) \\
    \midrule
''' + '\n'.join('    '+x for x in lines) + r'''
    \bottomrule
  \end{tabular}
  \par\smallskip
  \parbox{\columnwidth}{\footnotesize Endpoint RMSE is averaged equally over
  11 test flights. Tracking RMSE is averaged over three references and five
  paired initial conditions per reference.}
\end{table}
'''
    (ROOT.parent/'latex/tables/component_ablation.tex').write_text(table,encoding='utf-8')
    windows=pd.read_csv(OUT/'prediction_per_window.csv.gz').query('horizon_s == 0.5'); base=windows[windows.group=='full'][['flight_id','origin']].to_numpy()
    for g in GROUPS:
        np.testing.assert_array_equal(windows[windows.group==g][['flight_id','origin']].to_numpy(),base)
    assert len(base)==2515 and len(cases)==90
    write_json(OUT/'verification.json',dict(status='PASS',paired_primary_cases=75,matched_control_cases=15,new_closed_loop_cases=75,
        reused_closed_loop_cases=15,checked_initial_histories_and_references=True,checked_control_and_bias=True,
        primary_prediction_windows_per_group=len(base),verified_aggregate_sha256=checksum(OUT/'ablation_results.csv')))
    print(result.to_string())

if __name__=='__main__':
    main()
