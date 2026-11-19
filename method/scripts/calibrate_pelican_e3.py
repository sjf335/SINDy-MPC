"""Fit a restricted rigid-body simulator from E2 TRAIN flights only, then validate."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from pelican_e3.plant import derivative,step,allocation,steady_command
from pelican_e2.data import motor_estimate
from run_pelican_e2 import checksum,write_json
from verify_pelican_e2 import load_episode


def main(args):
    out = args.run_dir
    out.mkdir(parents=True,exist_ok=False)
    cfg = json.loads(args.config.read_text())
    e2 = ROOT/'runs/pelican_e2_v1'
    splits = json.loads((e2/'splits.json').read_text())
    motor = json.loads((e2/'motor_model.json').read_text())
    if args.motor_tau_scale != 1:
        motor['poles'] = (np.asarray(motor['poles'])**(1/args.motor_tau_scale)).tolist()
    started = time.monotonic()
    write_json(out/'config.json',cfg)
    write_json(out/'splits.json',splits)
    xall,yall,sall,weight,uall = [],[],[],[],[]
    training_hashes = {}
    for fid in splits['train']:
        e = load_episode(e2,fid)
        if args.motor_tau_scale != 1: e['motor_hat'] = motor_estimate(e['u'],motor)
        idx = np.arange(cfg['calibration_start'],len(e['x'])-1,cfg['calibration_stride'])
        x = e['x'][idx]
        r = Rotation.from_quat(x[:,6:10])
        v = r.inv().apply(x[:,3:6])
        omega = x[:,10:13]
        gravity = np.array([0,0,-cfg['gravity']])
        force = r.inv().apply((e['x'][idx+1,3:6]-x[:,3:6])/cfg['dt']-gravity)
        inertia = np.asarray(cfg['inertia_ratio_assumption'])
        alpha = (e['x'][idx+1,10:13]-omega)/cfg['dt']+np.cross(omega,omega*inertia)/inertia
        xall.append(np.c_[v,omega]); yall.append(np.c_[force,alpha])
        sall.append((e['motor_hat'][idx]/100)**2)
        weight.append(np.full(len(idx),np.sqrt(1/len(idx)/len(splits['train']))))
        uall.append(e['u'][idx])
        training_hashes[str(fid)] = checksum(e2/'data'/f'flight_{fid:02d}.npz')
    states,targets,speed,w,u = map(np.concatenate,[xall,yall,sall,weight,uall])
    drag = []
    for axis in range(2):
        v = states[:,axis]
        drag.append(max(0,float(np.sum(w*w*(-v)*targets[:,axis])/np.sum(w*w*v*v))))
    def regression(design,target,lower,upper):
        a,b = design*w[:,None],target*w
        rms = np.sqrt(np.maximum(np.sum(a*a,axis=0),1e-12))
        scaled = np.vstack([a/rms,np.eye(a.shape[1])*np.sqrt(cfg['calibration_ridge'])])
        rhs = np.r_[b,np.zeros(a.shape[1])]
        result = lsq_linear(scaled,rhs,bounds=(np.asarray(lower)*rms,np.asarray(upper)*rms),tol=1e-12,max_iter=100)
        assert result.success
        return result.x/rms
    thrust = regression(np.c_[speed,-states[:,2]],targets[:,2],[1e-6]*4+[0],[np.inf]*5)
    drag.append(thrust[-1])
    torque,angular_drag,aero_moment = [],[],[]
    for axis in range(3):
        design = np.c_[speed,-states[:,3+axis]]
        lower = [-np.inf]*4+[0]
        if args.aero_moments:
            design = np.c_[design,states[:,:3]]
            lower += [-np.inf]*3
        beta = regression(design,targets[:,3+axis],lower,[np.inf]*len(lower))
        torque.append(beta[:4]); angular_drag.append(beta[4])
        aero_moment.append(beta[5:] if args.aero_moments else np.zeros(3))
    gain,bias = np.asarray(motor['coefficients']).T
    parameters = {'kind':'independent_effective_rigid_body','gravity':cfg['gravity'],
                  'thrust_coefficients':thrust[:4],'linear_drag':drag,'torque_matrix':torque,
                  'angular_drag':angular_drag,'inertia_ratio':cfg['inertia_ratio_assumption'],
                  'aero_moment_matrix':aero_moment,'aero_moments_enabled':args.aero_moments,
                  'motor_tau':-cfg['dt']/np.log(motor['poles']),'motor_gain':gain,'motor_bias':bias,
                  'effective_motor_tau_scale':args.motor_tau_scale,
                  'training_flights':splits['train'],'training_transitions':len(states),
                  'training_data_sha256':training_hashes,'e1_motor_file_sha256':checksum(e2/'motor_model.json'),
                  'command_lower':np.quantile(u,cfg['command_quantiles'][0],axis=0),
                  'command_upper':np.quantile(u,cfg['command_quantiles'][1],axis=0),
                  'command_training_min':u.min(0),'command_training_max':u.max(0),
                  'assumptions':['Positive axial rotor thrust and diagonal nonnegative drag.',
                                 'Full effective torque allocation matrix; rotor geometry and absolute inertia not separately identified.',
                                 'Nominal inertia ratio 1:1:2, not measured Pelican inertia.',
                                 'Optional linear airspeed moments approximate aerodynamic moment coupling; coefficients are effective, not isolated blade-flapping constants.',
                                 'No sparse SINDy coefficient or test flight used in calibration.']}
    hover = steady_command(np.array([cfg['gravity'],0,0,0]),parameters)
    parameters['hover_command'] = hover
    parameters['allocation_condition_number'] = np.linalg.cond(allocation(parameters))
    parameters['hover_within_command_bounds'] = bool(np.all(hover>parameters['command_lower']) and np.all(hover<parameters['command_upper']))
    write_json(out/'plant_parameters.json',parameters)
    print(json.dumps({'stage':'fit_complete','thrust':thrust.tolist(),'drag':np.asarray(drag).tolist(),
                      'torque':np.asarray(torque).tolist(),'angular_drag':angular_drag,
                      'hover_command':hover.tolist(),'bounds':[parameters['command_lower'].tolist(),parameters['command_upper'].tolist()],
                      'allocation_condition':parameters['allocation_condition_number']},default=float),flush=True)
    rows = []
    for fid in splits['validation']:
        e = load_episode(e2,fid)
        if args.motor_tau_scale != 1: e['motor_hat'] = motor_estimate(e['u'],motor)
        origins = np.arange(200,len(e['x'])-101,200)
        initial = np.c_[e['x'][origins],e['motor_hat'][origins-1]]
        x = initial.copy()
        failed = np.zeros(len(origins),bool)
        for k in range(100):
            previous = x.copy()
            x = step(x,e['u'][origins+k],cfg['dt'],parameters,cfg['integration_dt'])
            bad = ~np.isfinite(x).all(1)|(np.linalg.norm(x[:,3:6],axis=1)>50)|(np.linalg.norm(x[:,10:13],axis=1)>100)
            failed |= bad
            x[failed] = initial[failed]
            if k+1 in [1,10,25,50,100]:
                true = e['x'][origins+k+1]
                pred = x[:,:13].copy()
                pred[:,3:6] = (x[:,:3]-previous[:,:3])/cfg['dt']
                pred[:,10:13] = (Rotation.from_quat(previous[:,6:10]).inv()*Rotation.from_quat(x[:,6:10])).as_rotvec()/cfg['dt']
                attitude = (Rotation.from_quat(true[:,6:10]).inv()*Rotation.from_quat(pred[:,6:10])).magnitude()
                ok = ~failed
                row = {'flight_id':fid,'horizon_s':(k+1)*cfg['dt'],'windows':len(origins),'failed_windows':int(failed.sum())}
                for name,slice_ in [('position',slice(0,3)),('velocity',slice(3,6)),('omega',slice(10,13))]:
                    row[name+'_rmse'] = float(np.sqrt(np.mean(np.sum((pred[ok,slice_]-true[ok,slice_])**2,axis=1))))
                row['attitude_rmse'] = float(np.sqrt(np.mean(attitude[ok]**2)))
                row['constant_velocity_rmse'] = float(np.sqrt(np.mean(np.sum((initial[:,3:6]-true[:,3:6])**2,axis=1))))
                rows.append(row)
        print(json.dumps({'stage':'validation_flight_complete','flight_id':fid}),flush=True)
    pd.DataFrame(rows).to_csv(out/'validation_per_flight.csv',index=False)
    summary = pd.DataFrame(rows).groupby('horizon_s').agg({'velocity_rmse':'mean','omega_rmse':'mean','position_rmse':'mean','attitude_rmse':'mean',
                                                       'constant_velocity_rmse':'mean','failed_windows':'sum','windows':'sum'}).reset_index()
    summary.to_csv(out/'validation_summary.csv',index=False)
    primary = summary[summary.horizon_s==.5].iloc[0]
    gates = {'hover_in_bounds':parameters['hover_within_command_bounds'],
             'allocation_nonsingular':parameters['allocation_condition_number']<1000,
             'validation_no_half_second_failure':primary.failed_windows==0,
             'validation_velocity_beats_constant':primary.velocity_rmse<primary.constant_velocity_rmse,
             'validation_half_second_rate_rmse_below_3':primary.omega_rmse<3}
    write_json(out/'CALIBRATED.json',{'utc':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-started,
                                    'gates':{k:bool(v) for k,v in gates.items()},'passed':bool(all(gates.values())),
                                    'plant_sha256':checksum(out/'plant_parameters.json'),
                                    'validation_summary':summary.to_dict(orient='records'),
                                    'calibration_script_sha256':checksum(Path(__file__))})
    print(summary.to_string(index=False)); print(json.dumps({k:bool(v) for k,v in gates.items()}),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=ROOT/'runs/pelican_e3_calibration_v1')
    p.add_argument('--config',type=Path,default=ROOT/'configs/pelican_e3.json')
    p.add_argument('--motor-tau-scale',type=float,default=1.)
    p.add_argument('--aero-moments',action='store_true')
    with threadpool_limits(limits=1): main(p.parse_args())
