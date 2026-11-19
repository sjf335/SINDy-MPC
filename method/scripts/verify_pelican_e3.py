"""Independently audit frozen E3 scenarios, physical transitions and metrics."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose,assert_array_equal
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from run_pelican_e2 import checksum,write_json


def reference_derivative(x,u,p):
    """Separate implementation using explicit rotation/mass-moment equations."""
    q=x[6:10]; w=x[10:13]; motor=x[13:17]
    matrix=Rotation.from_quat(q).as_matrix()
    velocity_body=matrix.T@x[3:6]
    actuator_squared=(motor/100.)**2
    force=-np.asarray(p['linear_drag'])*velocity_body
    force[2]+=np.dot(p['thrust_coefficients'],actuator_squared)
    rate=np.asarray(p['torque_matrix'])@actuator_squared
    rate+=np.asarray(p['aero_moment_matrix'])@velocity_body
    inertia=np.diag(p['inertia_ratio'])
    rate-=np.linalg.solve(inertia,np.cross(w,inertia@w))+np.asarray(p['angular_drag'])*w
    qx,qy,qz,qw=q
    kinematics=.5*np.array([[qw,-qz,qy],[qz,qw,-qx],[-qy,qx,qw],[-qx,-qy,-qz]])@w
    actuator=(np.asarray(p['motor_gain'])*u+np.asarray(p['motor_bias'])-motor)/np.asarray(p['motor_tau'])
    return np.r_[x[3:6],matrix@force+[0,0,-p['gravity']],kinematics,rate,actuator]


def verify(run):
    frozen=json.loads((run/'FROZEN.json').read_text())
    for path,expected in frozen['source_sha256'].items(): assert checksum(ROOT/path)==expected,path
    for path,expected in frozen['artifact_sha256'].items(): assert checksum(run/path)==expected,path
    completed=json.loads((run/'COMPLETED.json').read_text())
    assert completed['stage']=='test' and completed['completed_utc']>frozen['frozen_utc']
    cfg=json.loads((run/'config.json').read_text()); p=json.loads((run/'plant_parameters.json').read_text())
    qualification=json.loads((run/'development_qualification.json').read_text())
    development=ROOT/qualification['path']
    assert checksum(development/'tracking_per_run.csv')==qualification['metrics_sha256']
    assert checksum(development/'config.json')==qualification['config_sha256']
    assert checksum(development/'plant_parameters.json')==checksum(run/'plant_parameters.json')
    assert checksum(ROOT/frozen['calibration_source']/'plant_parameters.json')==checksum(run/'plant_parameters.json')
    split=json.loads((ROOT/'runs/pelican_e2_v1/splits.json').read_text())
    assert p['training_flights']==split['train']
    assert not (set(p['training_flights'])&set(split['test']))
    for fid,expected in p['training_data_sha256'].items():
        assert checksum(ROOT/'runs/pelican_e2_v1/data'/f'flight_{int(fid):02d}.npz')==expected
    assert checksum(run/'selected_models.json')==frozen['e2_selected_models_sha256']
    frame=pd.read_csv(run/'tracking_per_run.csv')
    expected={(k,s,m) for k in cfg['trajectories'] for s in cfg['test_seeds'] for m in cfg['controllers']}
    actual=set(zip(frame.trajectory,frame.seed,frame.controller))
    assert actual==expected and len(frame)==len(expected)==completed['cases']
    assert not frame.duplicated(['trajectory','seed','controller']).any()
    hold=round(cfg['control_dt']/cfg['dt']); nsteps=round(cfg['duration_s']/cfg['dt'])
    max_physical_error=0.; physical_checks=0; samples=0; paired={}
    for row in frame.itertuples(index=False):
        folder=run/'cases'/f'{row.trajectory}_{row.seed}_{row.controller}'
        with np.load(folder/'trajectory.npz') as f: data={key:f[key] for key in f.files}
        truth,obs,ref,u=data['physical'],data['observation'],data['reference'],data['command']
        assert len(truth)==len(u)+1 and truth.shape[1]==17 and obs.shape==(len(truth),13)
        assert_array_equal(data['t'],np.arange(len(truth))*cfg['dt'])
        assert_allclose(np.linalg.norm(truth[:,6:10],axis=1),1,atol=1e-12)
        assert_allclose(obs[:,:3],truth[:,:3],atol=0,rtol=0)
        assert_allclose(obs[:,6:10],truth[:,6:10],atol=0,rtol=0)
        assert_allclose(obs[1:,3:6],np.diff(truth[:,:3],axis=0)/cfg['dt'],atol=1e-12)
        rotations=Rotation.from_quat(truth[:,6:10])
        assert_allclose(obs[1:,10:13],(rotations[:-1].inv()*rotations[1:]).as_rotvec()/cfg['dt'],atol=1e-12)
        key=(row.trajectory,row.seed)
        if key in paired:
            other=paired[key]
            for name in ['initial_history_states','initial_history_commands']: assert_array_equal(data[name],other[name])
            assert_array_equal(truth[0],other['physical'][0])
            assert_array_equal(ref,other['reference'][:len(ref)]) if len(ref)<=len(other['reference']) else assert_allclose(ref[:len(other['reference'])],other['reference'],atol=0)
        else: paired[key]=data
        with np.load(run/'initial_conditions'/f'{row.seed}.npz') as initial:
            assert_array_equal(initial['states'],data['initial_history_states'])
            assert_array_equal(initial['commands'],data['initial_history_commands'])
            assert_array_equal(initial['truth'][-1],truth[0])
        errors=np.sqrt(np.sum((truth[:,:3]-ref[:,:3])**2,axis=1))
        rmse=float(np.sqrt(np.mean(errors*errors)))
        assert_allclose(row.prefix_position_rmse_m,rmse,atol=1e-12)
        assert_allclose(row.max_position_error_m,errors.max(),atol=1e-12)
        assert_allclose(row.final_position_error_m,errors[-1],atol=1e-12)
        is_complete=len(u)==nsteps and row.termination_reason=='completed'
        assert bool(row.complete)==is_complete
        if is_complete: assert_allclose(row.position_rmse_m,rmse,atol=1e-12)
        else: assert np.isnan(row.position_rmse_m)
        controls=pd.read_csv(folder/'control.csv')
        assert_array_equal(controls.step,np.arange(len(controls))*hold)
        assert_allclose(controls.t,controls.step*cfg['dt'],atol=1e-12)
        cmd=controls[[f'u{i}' for i in range(1,5)]].to_numpy()
        assert_allclose(u,np.repeat(cmd,hold,axis=0)[:len(u)],atol=1e-12)
        previous=np.vstack([data['initial_history_commands'][-1],cmd[:-1]])
        change=np.max(np.abs(cmd-previous),axis=1)
        assert_allclose(controls.max_command_change,change,atol=1e-12)
        bound_bad=np.any((u<np.asarray(p['command_lower'])-1e-7)|(u>np.asarray(p['command_upper'])+1e-7),axis=1)
        slew_bad=change>cfg['command_delta_per_control']+1e-7
        assert row.input_bound_violations==int(bound_bad.sum())
        assert row.input_slew_violations==int(slew_bad.sum())
        assert row.fallbacks==int(controls.fallback.sum()) and row.solver_failures==int((~controls.success).sum())
        assert row.control_calls==len(controls)
        success=is_complete and rmse<=cfg['tracking_success_rmse_m'] and errors.max()<=cfg['tracking_success_max_error_m'] and not bound_bad.any() and not slew_bad.any()
        assert bool(row.tracking_success)==bool(success)
        assert_allclose(row.command_variation_squared,np.sum(np.diff(cmd,axis=0)**2),atol=1e-9)
        assert_allclose(row.command_squared_integral_code_units,np.sum(u*u)*cfg['dt'],rtol=1e-12)
        assert_allclose(row.solve_time_p95_s,np.quantile(controls.seconds,.95),atol=1e-12)
        assert_allclose(row.controller_time_p95_s,np.quantile(controls.controller_wall_s,.95),atol=1e-12)
        assert_allclose(row.controller_over_period_fraction,np.mean(controls.controller_wall_s>cfg['control_dt']),atol=1e-12)
        # Independent adaptive integration on six spread-out transitions in every case.
        for k in np.unique(np.linspace(0,len(u)-1,6).astype(int)):
            adaptive=solve_ivp(lambda t,x:reference_derivative(x,u[k],p),[0,cfg['dt']],truth[k],rtol=1e-11,atol=1e-12).y[:,-1]
            adaptive[6:10]/=np.linalg.norm(adaptive[6:10])
            difference=float(np.max(np.abs(adaptive-truth[k+1])))
            assert_allclose(adaptive,truth[k+1],atol=5e-8,rtol=5e-8)
            max_physical_error=max(max_physical_error,difference); physical_checks+=1
        samples+=len(u)
    summary=pd.read_csv(run/'tracking_summary.csv')
    for row in summary.itertuples(index=False):
        subset=frame[(frame.trajectory==row.trajectory)&(frame.controller==row.controller)]
        assert row.runs==len(subset) and row.completed==int(subset.complete.sum())
        assert row.tracking_successes==int(subset.tracking_success.sum())
        assert row.fallbacks==int(subset.fallbacks.sum()) and row.solver_failures==int(subset.solver_failures.sum())
        assert_allclose(row.position_rmse_mean_successful_completion_only,subset.position_rmse_m.mean(),atol=1e-12,equal_nan=True)
    result={'status':'PASS','verified_utc':datetime.now(timezone.utc).isoformat(),'cases':len(frame),
            'paired_scenarios':len(paired),'physical_transitions_logged':samples,
            'independent_adaptive_transitions_checked':physical_checks,'max_independent_transition_difference':max_physical_error,
            'frozen_hashes_valid':True,'e2_models_unchanged':True,'plant_calibration_train_only':True,
            'paired_initial_histories_identical':True,'metrics_recomputed_from_saved_trajectories':True,
            'control_hold_and_constraints_checked':True,'verifier_sha256':checksum(Path(__file__)),
            'limitations':['Same-agent independent code path; not an external audit.',
                           'No repeated full optimization run; no real-time timing reproducibility claim.',
                           'Verification validates implementation and recorded evidence, not physical fidelity to the real Pelican.']}
    write_json(run/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/pelican_e3_v1')
    verify(parser.parse_args().run_dir)
