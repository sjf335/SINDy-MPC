"""Independently check frozen E2 rollouts and recompute saved test metrics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from pelican_e2.model import context_for_episode, rollout
from run_pelican_e2 import checksum, restore_model, write_json


def load_episode(run, fid):
    with np.load(run/'data'/f'flight_{fid:02d}.npz') as f:
        return {'id': fid, **{key: f[key] for key in f.files}}


def independent_rollout(episode, origin, model, motor, cfg):
    """Scalar reference: SciPy rotations, explicit monomials, separate history.

    Does not call production feature construction, rotation, advance or rollout.
    The only true states read are the prefix up to and including the origin.
    """
    dt = cfg['dt']
    steps = round(max(cfg['horizons_s'])/dt)
    gravity = np.array([0., 0., -cfg['gravity']])
    state = episode['x'][origin].copy()
    trajectory = np.full((steps+1, 13), np.nan)
    trajectory[0] = state
    failed_at = steps+1
    constant = model['family'] == 'constant_twist'
    if not constant:
        def z_value(x, u):
            r = Rotation.from_quat(x[6:10])
            z = np.r_[r.inv().apply(x[3:6]), x[10:13], r.inv().apply(gravity), u]
            return (z-model['mu'])/model['scale']
        input_key = 'motor_hat' if model['motor'] else 'u'
        history = {k-origin: z_value(episode['x'][k], episode[input_key][k])
                   for k in range(origin-max(model['lags'], default=0), origin+1)}
        estimated_motor = episode['motor_hat'][origin-1].copy()
        poles = np.asarray(motor['poles'])
        coefficients = np.asarray(motor['coefficients'])
    for step in range(steps):
        r = Rotation.from_quat(state[6:10])
        updated = state.copy()
        if constant:
            updated[:3] += dt*state[3:6]
            updated[6:10] = (r*Rotation.from_rotvec(dt*state[10:13])).as_quat()
        else:
            u = episode['u'][origin+step]
            if model['motor']:
                estimated_motor = poles*estimated_motor+(1-poles)*(coefficients[:, 0]*u+coefficients[:, 1])
                u = estimated_motor
            z = z_value(state, u)
            history[step] = z
            phi = [1., *z]
            if model['degree'] == 2:
                phi.extend(z[i]*z[j] for i in range(13) for j in range(i, 13))
            for lag in model['lags']:
                before = history[step-lag]
                phi.extend(before[i]-z[i] for i in [0,1,2,3,4,5,9,10,11,12])
                if model['degree'] == 2:
                    phi.extend(before[i]**2-z[i]**2 for i in range(9, 13))
            target = model['coefficients']@(np.asarray(phi)/model['feature_scale'])
            target = target*model['y_scale']+model['y_mu']
            updated[3:6] += dt*(r.apply(target[:3])+gravity)
            updated[10:13] += dt*target[3:]
            updated[:3] += dt*updated[3:6]
            updated[6:10] = (r*Rotation.from_rotvec(dt*updated[10:13])).as_quat()
        valid = (np.isfinite(updated).all()
                 and np.linalg.norm(updated[3:6]) <= cfg['max_speed_m_s']
                 and np.linalg.norm(updated[10:13]) <= cfg['max_body_rate_rad_s'])
        if not valid:
            failed_at = step+1
            break
        state = updated
        trajectory[step+1] = state
    return trajectory, failed_at


def verify_freeze(run):
    freeze = json.loads((run/'FROZEN.json').read_text())
    for path, expected in freeze['code_sha256'].items():
        assert checksum(ROOT/path) == expected, path
    for key, path in [('selected_models_sha256','models/selected_models.json'),
                      ('config_sha256','config.json'), ('splits_sha256','splits.json'),
                      ('motor_model_sha256','motor_model.json')]:
        assert checksum(run/path) == freeze[key], path
    assert checksum(ROOT/'data/pelican/raw/AscTec_Pelican_Flight_Dataset.mat') == freeze['raw_sha256']
    assert checksum(ROOT/'runs/pelican_e1_v1/motor_response_models.json') == freeze['e1_motor_models_sha256']
    splits = json.loads((run/'splits.json').read_text())
    sets = [set(splits[k]) for k in ['train','validation','test']]
    assert [len(x) for x in sets] == [32,11,11]
    assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
    assert set.union(*sets) == set(range(1,55))
    e1 = json.loads((ROOT/'runs/pelican_e1_v1/splits.json').read_text())
    assert all(splits[k] == e1[k] for k in ['train','validation','test'])
    return freeze, splits


def compare_reference(run, ids, models, motor, cfg, final=False):
    rows = []
    nsteps = round(max(cfg['horizons_s'])/cfg['dt'])
    for model in models:
        endpoints = np.load(run/'test'/f'{model["name"]}_endpoints.npz') if final else None
        for fid in ids:
            e = load_episode(run, fid)
            available = np.arange(cfg['origin_start_index'], len(e['x'])-nsteps, cfg['origin_stride'])
            origins = available[[0, len(available)//2]]
            initial, history, prior = context_for_episode(e, origins, model, cfg)
            inputs = e['u'][origins[:, None]+np.arange(nsteps)]
            _, failure, path = rollout(initial, history, inputs, prior, model, motor, cfg, keep_all=True)
            for j, origin in enumerate(origins):
                reference, ref_failure = independent_rollout(e, int(origin), model, motor, cfg)
                # Quaternion antipodes denote the same rotation.
                sign = np.where(np.sum(reference[:,6:10]*path[j,:,6:10], axis=1)<0,-1,1)
                reference[:,6:10] *= sign[:,None]
                assert failure[j] == ref_failure
                assert_allclose(path[j], reference, rtol=2e-8, atol=2e-8, equal_nan=True)
                max_error = float(np.nanmax(np.abs(path[j]-reference)))
                if final:
                    where = int(np.flatnonzero(available == origin)[0])
                    for h in cfg['horizons_s']:
                        step = round(h/cfg['dt'])
                        prefix = f'f{fid:02d}_s{step:03d}'
                        assert_allclose(endpoints[prefix+'_prediction'][where], path[j,step],
                                        rtol=2e-8, atol=2e-8, equal_nan=True)
                        assert endpoints[prefix+'_failed_at_step'][where] == ref_failure
                rows.append({'model':model['name'],'flight_id':fid,'origin':int(origin),
                             'max_absolute_difference':max_error,'failed_at_step':ref_failure})
            # A deliberately corrupted future state/motor record cannot alter the
            # initialized context or predictions. Future recorded commands are allowed.
            origin = int(origins[0])
            changed = {k: v.copy() if isinstance(v,np.ndarray) else v for k,v in e.items()}
            changed['x'][origin+1:] = np.nan
            changed['motor_hat'][origin+1:] = np.nan
            before = context_for_episode(e,np.array([origin]),model,cfg)
            after = context_for_episode(changed,np.array([origin]),model,cfg)
            for a,b in zip(before,after): assert_array_equal(a,b)
        if endpoints is not None: endpoints.close()
    return rows


def verify_metrics(run, ids, models, cfg):
    frame = pd.read_csv(run/'test/prediction_per_flight.csv')
    windows = pd.read_csv(run/'test/prediction_per_window.csv')
    summaries = pd.read_csv(run/'test/prediction_summary.csv')
    assert len(frame) == len(models)*len(ids)*len(cfg['horizons_s'])
    assert not frame.duplicated(['model','flight_id','horizon_s']).any()
    assert not windows.duplicated(['model','flight_id','horizon_s','origin_index']).any()
    checked_windows = 0
    max_metric_difference = 0.
    for model in models:
        with np.load(run/'test'/f'{model["name"]}_endpoints.npz') as data:
            for fid in ids:
                e = load_episode(run,fid)
                assert_array_equal(e['raw_index'],np.arange(1,len(e['x'])+1))
                assert_allclose(e['t'],e['raw_index']*cfg['dt'],atol=0,rtol=0)
                for horizon in cfg['horizons_s']:
                    step = round(horizon/cfg['dt'])
                    prefix = f'f{fid:02d}_s{step:03d}'
                    pred,truth = data[prefix+'_prediction'],data[prefix+'_truth']
                    origins,failed = data[prefix+'_origins'],data[prefix+'_failed_at_step']
                    assert_array_equal(truth,e['x'][origins+step])
                    ok = failed>step
                    assert np.isfinite(pred[ok]).all() and np.isnan(pred[~ok]).all()
                    assert_allclose(np.linalg.norm(pred[ok,6:10],axis=1),1,atol=1e-12)
                    row = frame[(frame.model==model['name'])&(frame.flight_id==fid)&(frame.horizon_s==horizon)].iloc[0]
                    assert row.windows == len(origins) and row.failed_windows == int((~ok).sum())
                    assert_allclose(row.failure_rate,(~ok).mean(),atol=1e-14)
                    subset = windows[(windows.model==model['name'])&(windows.flight_id==fid)&(windows.horizon_s==horizon)]
                    assert_array_equal(subset.origin_index,origins)
                    assert_array_equal(subset.raw_origin_index,e['raw_index'][origins])
                    assert_array_equal(subset.failed,~ok)
                    for metric,columns in [('position',slice(0,3)),('velocity',slice(3,6)),('omega',slice(10,13)),('attitude',None)]:
                        errors = np.full(len(origins),np.nan)
                        if metric == 'attitude':
                            errors[ok] = (Rotation.from_quat(truth[ok,6:10]).inv()*Rotation.from_quat(pred[ok,6:10])).magnitude()
                        else:
                            errors[ok] = np.sqrt(np.sum((pred[ok,columns]-truth[ok,columns])**2,axis=1))
                        rmse = np.sqrt(np.mean(errors[ok]**2)) if ok.any() else np.nan
                        assert_allclose(row[metric+'_rmse'],rmse,atol=5e-8,rtol=1e-9,equal_nan=True)
                        assert_allclose(subset[metric+'_error'],errors,atol=5e-8,rtol=1e-9,equal_nan=True)
                        max_metric_difference = max(max_metric_difference,abs(row[metric+'_rmse']-rmse))
                    checked_windows += len(origins)
        for horizon in cfg['horizons_s']:
            a = frame[(frame.model==model['name'])&(frame.horizon_s==horizon)]
            b = summaries[(summaries.model==model['name'])&(summaries.horizon_s==horizon)].iloc[0]
            assert b.test_flights == len(ids) and b.windows == a.windows.sum()
            assert b.failed_windows == a.failed_windows.sum()
            assert bool(b.complete_all_flights) == bool((a.failed_windows==0).all())
            assert_allclose(b.mean_flight_failure_rate,a.failure_rate.mean(),atol=1e-14)
            for metric in ['position','velocity','omega','attitude']:
                assert_allclose(b[metric+'_rmse_mean'],a[metric+'_rmse'].mean(),atol=1e-12)
    assert checked_windows == len(windows)
    with np.load(run/'test/linear_history_endpoints.npz') as a, np.load(run/'test/linear_history_matched_endpoints.npz') as b:
        for key in a.files: assert_array_equal(a[key],b[key])
    return {'per_flight_rows':len(frame),'window_endpoint_rows':checked_windows,
            'summary_rows':len(summaries),'max_metric_difference':max_metric_difference,
            'matched_linear_identical':True}


def main(args):
    run = args.run_dir
    freeze,splits = verify_freeze(run)
    config = json.loads((run/'config.json').read_text())
    motor = json.loads((run/'motor_model.json').read_text())
    models = [restore_model(m) for m in json.loads((run/'models/selected_models.json').read_text())]
    final = args.stage == 'final'
    ids = splits['test'] if final else splits['validation'][:3]
    if not final:
        assert not list((run/'test').iterdir()), 'Pretest check requires an untouched test directory'
        assert all(not (run/'data'/f'flight_{fid:02d}.npz').exists() for fid in splits['test'])
    else:
        completed = json.loads((run/'test/COMPLETED.json').read_text())
        assert completed['models_frozen_utc'] == freeze['frozen_utc']
        assert completed['completed_utc'] > freeze['frozen_utc']
        events = [json.loads(line) for line in (run/'progress.jsonl').read_text().splitlines()]
        load_events = [e for e in events if e['event']=='frozen_models_verified_before_test_load']
        assert len(load_events)==1 and load_events[0]['utc'] > freeze['frozen_utc']
    checks = compare_reference(run,ids,models,motor,config,final)
    result = {'status':'PASS','stage':args.stage,'verified_utc':datetime.now(timezone.utc).isoformat(),
              'frozen_hashes_valid':True,'e1_splits_preserved':True,
              'future_state_isolation_all_models':True,'reference_rollouts':len(checks),
              'reference_max_absolute_difference':max(x['max_absolute_difference'] for x in checks),
              'reference_details':checks,'verifier_sha256':checksum(Path(__file__))}
    if final: result['metric_recomputation'] = verify_metrics(run,ids,models,config)
    filename = 'verification.json' if final else 'pretest_verification.json'
    write_json(run/filename,result)
    print(json.dumps({k:v for k,v in result.items() if k!='reference_details'},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['pretest','final'],required=True)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/pelican_e2_v1')
    with threadpool_limits(limits=1): main(parser.parse_args())
