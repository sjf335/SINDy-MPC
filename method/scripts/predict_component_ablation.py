"""Score all five prediction variants on identical E2 test windows with causal bias."""
from run_component_ablation import E2, TRACK, OUT, GROUPS, get_model, read, checksum, write_json
import numpy as np
import pandas as pd
from pelican_e2.data import state_features, target_acceleration
from pelican_e2.model import context_for_episode, library, origins_for, rollout
from pelican_e3.control import pack_observation
from pelican_e3_corrected.control import BiasObserver, symbolic_transition


def bias_history(episode, model, cfg, settings, start):
    """estimate[i] uses x through i and commands through i-1, for every group."""
    x, u = episode['x'], episode['u']
    z = (np.c_[state_features(x,cfg['gravity']), u] - model['mu']) / model['scale']
    gate = np.array(settings['residual_gate']); limit = np.array(settings['estimate_limit'])
    alpha = -np.expm1(-cfg['dt'] / np.array(settings['time_constant_s']))
    estimate = np.zeros_like(x[:, :6]); state = np.zeros(6)
    for chunk in range(start, len(x)-1, 4096):
        idx = np.arange(chunk, min(chunk+4096,len(x)-1))
        phi = library(z[idx], {lag:z[idx-lag] for lag in model['lags']}, model['lags'], model['degree'])
        predicted = ((phi / model['feature_scale']) @ model['coefficients'].T) * model['y_scale'] + model['y_mu']
        innovations = target_acceleration(episode, idx, cfg) - predicted
        accepted = np.all(np.isfinite(innovations) & (np.abs(innovations) <= gate),axis=1)
        for i, innovation, valid in zip(idx, innovations, accepted):
            if valid:
                state = np.clip(state + alpha*(innovation-state), -limit, limit)
            estimate[i+1] = state
    return estimate


def verify_scalar(episode, model, cfg, motor, observer, start, origins, bias, predictions):
    x,u = episode['x'],episode['u']; transition,_ = symbolic_transition(model,motor,cfg)
    def packed(i):
        return pack_observation(x[:i+1],u[:i],episode['motor_hat'][i-1],model,cfg)
    if observer:
        live = BiasObserver(transition,cfg['dt'],observer)
        for i in range(start,origins[0]):
            _,est,_,_ = live.update(packed(i),u[i],x[i+1]); np.testing.assert_allclose(est,bias[i+1],atol=1e-10,rtol=1e-10)
    for j in [0,len(origins)//2]:
        i = origins[j]; s = packed(i)
        for k in range(50):
            s = np.asarray(transition(s,u[i+k],bias[i])).ravel()
        np.testing.assert_allclose(s[:13], predictions[50][j],atol=3e-8,rtol=1e-8)


def main():
    cfg = read(E2/'config.json'); control_cfg = read(TRACK/'selected_configs/circle.json')
    ids = read(E2/'splits.json')['test']; motor = read(TRACK/'motor_model.json'); obs = read(TRACK/'observer.json')
    assert len(ids)==11 and cfg['dt']==control_cfg['dt']
    models = {g:get_model(g,control_cfg) for g in GROUPS}; start=max(max(m['lags'],default=0) for m in models.values())
    win_rows = []; used = {}
    for fid in ids:
        path=E2/'data'/f'flight_{fid:02}.npz'; used[path.name]=checksum(path)
        with np.load(path) as f: e=dict(f)
        origins = origins_for(e,cfg); steps=round(max(cfg['horizons_s'])/cfg['dt']); inputs=e['u'][origins[:,None]+np.arange(steps)]
        for group,m in models.items():
            online=GROUPS[group][2]; bias=bias_history(e,m,cfg,obs,start) if online else np.zeros_like(e['x'][:,:6])
            x,h,prior=context_for_episode(e,origins,m,cfg)
            # Broadcast an individual, fixed output correction per window over its horizon.
            effective={**m,'y_mu':m['y_mu']+bias[origins]}
            predictions,failed,_=rollout(x,h,inputs,prior,effective,motor,cfg)
            verify_scalar(e,m,cfg,motor,obs if online else None,start,origins,bias,predictions)
            for horizon in cfg['horizons_s']:
                n=round(horizon/cfg['dt']); pred=predictions[n]; truth=e['x'][origins+n]; dot=(pred[:,6:10]*truth[:,6:10]).sum(axis=1)
                win_rows.append(pd.DataFrame(dict(group=group,flight_id=fid,origin=origins,horizon_s=horizon,failed=failed<=n,
                    position_m=np.linalg.norm(pred[:,:3]-truth[:,:3],axis=1),velocity_m_s=np.linalg.norm(pred[:,3:6]-truth[:,3:6],axis=1),
                    attitude_rad=2*np.arccos(np.clip(abs(dot),0,1)),omega_rad_s=np.linalg.norm(pred[:,10:13]-truth[:,10:13],axis=1))))
        print('scored flight',fid,flush=True)
    all_windows=pd.concat(win_rows,ignore_index=True); by=['group','flight_id','horizon_s']; metrics=['position_m','velocity_m_s','attitude_rad','omega_rad_s']
    assert not all_windows.duplicated(by+['origin']).any(); counts=all_windows.groupby(by).failed.agg(windows='size',failed='sum')
    flight=all_windows.assign(**{k:all_windows[k]**2 for k in metrics}).groupby(by)[metrics].mean().pow(.5).join(counts).reset_index()
    g=flight.groupby(['group','horizon_s']); summary=g[metrics].mean().join(g[['windows','failed']].sum()).reset_index()
    assert (summary.windows==2515).all() and len(flight)==275
    all_windows.to_csv(OUT/'prediction_per_window.csv.gz',index=False); flight.to_csv(OUT/'prediction_per_flight.csv',index=False);summary.to_csv(OUT/'prediction_summary.csv',index=False)
    write_json(OUT/'prediction_verification.json',dict(files_sha256=used,script_sha256=checksum(__file_path()), flights=len(ids),common_start=start,common_windows=2515,
        scalar_observer_and_rollout='PASS',model_counts={g:[m['feature_count'],m['nonzero_coefficients']] for g,m in models.items()}))


def __file_path():
    from pathlib import Path
    return Path(__file__)


if __name__=='__main__':
    main()
