"""Recalculate paper intervals from paired flight and simulation records."""
import argparse
from itertools import product
from math import comb
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
E2 = RUNS / 'pelican_e2_v1'
TRACK = RUNS / 'pelican_trajectory_search_v1'
E4 = RUNS / 'pelican_e4_v1'
CANDIDATE = 'sindy_history'
BASELINES = ('pd', 'sindy_current', 'linear_history', 'motor_sindy')


def effect(baseline, candidate, draws):
    delta = baseline - candidate
    b = baseline[draws].mean(axis=1)
    d = delta[draws].mean(axis=1)
    absolute = np.quantile(d, [.025, .975]); relative = np.quantile(100*d/b, [.025, .975])
    n = int(np.count_nonzero(delta)); wins = int((delta > 0).sum())
    sign_p = min(1., 2*sum(comb(n,k) for k in range(min(wins,n-wins)+1))/2**n)
    return dict(baseline_mean=baseline.mean(), candidate_mean=candidate.mean(), improvement=delta.mean(),
                lower95=absolute[0], upper95=absolute[1], improvement_pct=100*delta.mean()/baseline.mean(),
                lower95_pct=relative[0], upper95_pct=relative[1], positive_blocks=wins,
                nonzero_blocks=n, sign_p_two_sided=sign_p)


def predictions():
    cfg=json.loads((E2/'config.json').read_text()); splits=json.loads((E2/'splits.json').read_text()); ids=sorted(splits['test'])
    assert len(ids)==11 and len(set(splits['train'] + splits['validation'] + ids))==54
    windows=pd.read_csv(E2/'test/prediction_per_window.csv'); flight=pd.read_csv(E2/'test/prediction_per_flight.csv')
    keys=['model','flight_id','horizon_s']; metrics=['velocity','position','attitude','omega']
    assert not windows.duplicated(keys+['origin_index']).any() and windows.failed.dtype==bool
    assert windows.groupby(['model','horizon_s']).size().eq(2515).all()
    # Check all recorded endpoint errors against every flight-level RMSE and failure count.
    for metric in metrics:
        errors=windows[metric+'_error']; assert errors.isna().equals(windows.failed)
        values=windows.assign(squared=errors**2).groupby(keys).squared.mean().pow(.5)
        expected=flight.set_index(keys)[metric+'_rmse'].sort_index()
        pd.testing.assert_index_equal(values.index,expected.index)
        np.testing.assert_allclose(values,expected,atol=2e-12,rtol=1e-11)
    np.testing.assert_array_equal(windows.groupby(keys).failed.sum(),flight.set_index(keys).sort_index().failed_windows)
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    draws=rng.integers(0,len(ids),size=(cfg['bootstrap_repetitions'],len(ids)))
    rows=[]; half=flight[flight.horizon_s==cfg['primary_horizon_s']]; assert not half.failed_windows.any()
    for baseline in ('sindy_current','linear_history','motor_sindy','constant_twist'):
        for metric in metrics:
            pivot=half.pivot(index='flight_id',columns='model',values=metric+'_rmse').loc[ids]
            rows.append(dict(comparison='prediction_0.5s',reference='all',baseline=baseline,candidate=CANDIDATE,
                             metric=metric,unit={'velocity':'m/s','position':'m','omega':'rad/s','attitude':'rad'}[metric],blocks=len(ids),resamples=len(draws),complete=True,
                             **effect(pivot[baseline].to_numpy(),pivot[CANDIDATE].to_numpy(),draws)))
    return rows


def simulate_contrasts(frame, controller_key, candidate, baselines, seed_ids, trajectories, metrics, condition):
    assert set(frame[controller_key])==set(baselines)|{candidate}
    expected=pd.MultiIndex.from_product([seed_ids,trajectories],names=['seed','trajectory'])
    draws=np.array(list(product(range(len(seed_ids)),repeat=len(seed_ids))))
    rows=[]
    for col,scale in metrics.items():
        pivot=frame.pivot(index=['seed','trajectory'],columns=controller_key,values=col).sort_index()
        pd.testing.assert_index_equal(pivot.index,expected.sort_values())
        for label in (*trajectories,'all'):
            selected=pivot if label=='all' else pivot.loc[pivot.index.get_level_values('trajectory')==label]
            for baseline in baselines:
                pair=selected[[baseline,candidate]]; valid=pair.notna().all(axis=1)
                record=dict(comparison=condition,reference=label,baseline=baseline,candidate=candidate,metric=col,
                            unit='s' if col in ('recovery_time_s','recovery_s') else 'cm',scale_from_input=scale,blocks=len(seed_ids),resamples=len(draws),pair_count=len(pair),
                            valid_pairs=int(valid.sum()),complete=bool(valid.all()))
                if valid.all():
                    means=scale*pair.groupby('seed').mean().loc[seed_ids]
                    record.update(effect(means[baseline].to_numpy(),means[candidate].to_numpy(),draws))
                rows.append(record)
    return rows


def main(out):
    out.mkdir(parents=True,exist_ok=True)
    rows=predictions(); tracks=pd.read_csv(TRACK/'validate/tracking_per_run.csv')
    assert len(tracks)==75 and tracks.complete.all()
    track_seeds=list(range(24301,24306)); conditions=['circle','figure8','helix']
    rows+=simulate_contrasts(tracks,'controller',CANDIDATE,BASELINES,track_seeds,conditions,{'position_rmse_m':100},'nominal_tracking')
    e4=pd.read_csv(E4/'robustness_per_run.csv'); assert len(e4)==250
    robust_seeds=list(range(15301,15306))
    for condition,frames in e4.groupby('scenario'):
        assert len(frames)==50
        assert frames.position_rmse_m.notna().equals(frames.complete)
        m={'position_rmse_m':100,'late_rmse_m':100}
        if condition=='force_pulse':
            assert frames.recovery_time_s.notna().equals(frames.recovered)
            m.update(event_peak_error_m=100,post_event_rmse_m=100,recovery_time_s=1)
        rows+=simulate_contrasts(frames,'controller',CANDIDATE,BASELINES,robust_seeds,conditions[:2],m,'E4_'+condition)
    ablation=pd.read_csv(RUNS/'pelican_component_ablation_v1/tracking_per_case.csv')
    rows+=simulate_contrasts(ablation,'group','full',sorted(set(ablation.group)-{'full'}),track_seeds,conditions,
                             {'rmse_cm':1},'component_ablation')
    pulses=pd.read_csv(RUNS/'pelican_noise_pulse_v1/pulse_per_case.csv')
    for group,kinds in [('pulse',conditions[:2]),('short',['helix'])]:
        sub=pulses[pulses.arm.str.startswith(group+'_')]
        rows+=simulate_contrasts(sub,'arm',group+'_on',[group+'_off',group+'_fast'],robust_seeds,kinds,
                                 {'event_peak_cm':1,'recovery_s':1,'late_rmse_cm':1},group+'_response')
    result=pd.DataFrame(rows); result.to_csv(out/'paired_intervals.csv',index=False,float_format='%.12g')
    selected=result[(result.reference=='all') & (((result.comparison=='prediction_0.5s') & (result.baseline=='sindy_current') & result.metric.isin(['velocity','position'])) |
        ((result.comparison=='nominal_tracking') & result.baseline.isin(['pd','linear_history'])) |
        ((result.comparison=='E4_tau_2') & (result.baseline=='linear_history') & (result.metric=='late_rmse_m')) |
        ((result.comparison=='E4_force_pulse') & (result.baseline=='linear_history') & (result.metric=='post_event_rmse_m')))]
    assert len(selected)==6
    selected.to_csv(out/'paper_contrasts.csv',index=False,float_format='%.12g')
    print(selected[['comparison','baseline','metric','improvement','lower95','upper95','improvement_pct','lower95_pct','upper95_pct','sign_p_two_sided']].round(4).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=RUNS/'pelican_statistics_v1')
    main(parser.parse_args().out)
