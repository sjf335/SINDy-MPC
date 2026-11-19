"""Replay the Pelican model grid, the real-flight predictions and one nominal closed loop."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from run_pelican_e2 import ROOT, checksum, restore_model, write_json
from pelican_e2.data import prepare_flight
from pelican_e2.model import fit_candidate, evaluate
from pelican_e3_corrected.control import BiasMPC
from run_pelican_e3_nominal_comparison import run_case

E2 = ROOT / 'runs/pelican_e2_v1'
TRACK = ROOT / 'runs/pelican_trajectory_search_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def refit_grid():
    cfg = read(E2/'config.json')
    scores = pd.read_csv(E2/'validation/model_selection.csv')
    worst = 0.
    with np.load(E2/'models/training_statistics.npz') as sufficient:
        for record in scores.itertuples(index=False):
            original = restore_model(read(E2/'models'/f'{record.candidate}.json'))
            kind = 'motor' if original['motor'] else 'command'
            scales = {key: original[key] for key in ('mu','scale','y_mu','y_scale')}
            fit = fit_candidate(sufficient[kind+'_gram'], sufficient[kind+'_rhs'], scales,
                                json.loads(record.lags), record.degree, record.alpha, record.threshold, cfg, original['motor'])
            np.testing.assert_allclose(fit['coefficients'], original['coefficients'], rtol=1e-7, atol=1e-9)
            worst = max(worst, np.abs(fit['coefficients']-original['coefficients']).max())
            assert fit['nonzero_coefficients'] == original['nonzero_coefficients']
    for original in read(E2/'models/selected_models.json'):
        if original['name'] in ('constant_twist','linear_history_matched'):
            continue
        ranked = scores[scores.family==original['family']].sort_values(
            ['primary_failure_rate','primary_velocity_rmse','nonzero_coefficients'], kind='stable')
        assert ranked.iloc[0].candidate == original['candidate_name']
    return dict(refitted_candidates=len(scores), max_coefficient_difference=worst)


def replay_flights(raw_file):
    provenance = read(ROOT/'runs/pelican_e1_v1/raw_provenance.json')
    if checksum(raw_file) != provenance['sha256']:
        raise ValueError('Raw MAT differs from the flight archive used for the paper')
    raw=loadmat(raw_file,simplify_cells=True)['flights']
    cfg=read(E2/'config.json')
    motor=read(E2/'motor_model.json')
    flights=[prepare_flight(raw[fid-1],fid,cfg['dt'],motor) for fid in read(E2/'splits.json')['test']]
    reference=pd.read_csv(E2/'test/prediction_per_flight.csv').set_index(['model','flight_id','horizon_s']).sort_index()
    rows=[]
    for model in read(E2/'models/selected_models.json'):
        scores,_,_,_=evaluate(restore_model(model),flights,motor,cfg)
        rows.extend(scores)
    actual=pd.DataFrame(rows).set_index(reference.index.names).sort_index()
    pd.testing.assert_index_equal(reference.index,actual.index)
    fields=['velocity_rmse','position_rmse','omega_rmse','attitude_rmse']
    np.testing.assert_allclose(actual[fields],reference[fields],rtol=1e-7,atol=1e-9)
    np.testing.assert_array_equal(actual[['windows','failed_windows']],reference[['windows','failed_windows']])
    return dict(flights=len(flights),model_flight_horizons=len(actual),raw_sha256=provenance['sha256'])


def replay_closed_loop(destination):
    cfg=read(TRACK/'selected_configs/circle.json')
    model=restore_model(read(TRACK/'corrected_models.json')['sindy_history'])
    motor,plant=read(TRACK/'motor_model.json'),read(TRACK/'plant_parameters.json')
    seed=24301
    with np.load(TRACK/'validate/initial_conditions'/f'{seed}.npz') as data:
        initial=dict(data)
    settings=dict(observer=read(TRACK/'observer.json'),stage_runtime_limit_s=3600)
    result=run_case('sindy_history','circle',seed,BiasMPC(model,motor,plant,cfg),initial,plant,motor,cfg,
                    settings,destination,lambda *args,**kwargs:None,time.monotonic())
    frozen=pd.read_csv(TRACK/'validate/tracking_per_run.csv')
    expected=frozen.query("controller=='sindy_history' and trajectory=='circle' and seed==24301").iloc[0]
    for field in ('position_rmse_m','late_rmse_m','max_position_error_m'):
        np.testing.assert_allclose(result[field],expected[field],rtol=2e-6,atol=1e-8)
    assert result['complete'] and result['fallbacks']==expected.fallbacks==0
    return dict(seed=seed,calls=result['control_calls'],position_rmse_cm=100*result['position_rmse_m'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat',type=Path,help='Original Pelican MAT file for the test-flight replay')
    parser.add_argument('--out',type=Path,required=True,help='New output directory')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    with threadpool_limits(limits=1):
        report={'grid':refit_grid(),'closed_loop':replay_closed_loop(args.out/'circle')}
        if args.mat:
            report['prediction']=replay_flights(args.mat)
    write_json(args.out/'replay.json',report)
    print(json.dumps(report,indent=2))
