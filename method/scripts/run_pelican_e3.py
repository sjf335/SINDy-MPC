"""Develop, freeze, then run paired simulated closed-loop Pelican E3 scenarios."""
import argparse
from datetime import datetime,timezone
import json
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from pelican_e3.plant import step,observe
from pelican_e3.control import MPC,pd_command,bounded_command,pack_observation
from pelican_e3.reference import at
from run_pelican_e2 import checksum,restore_model,write_json


def initialize(seed,parameters,motor,cfg):
    rng = np.random.default_rng(seed)
    hover = np.asarray(parameters['hover_command'])
    initial = np.r_[np.array([0.,0.,1.])+rng.normal(0,cfg['initial_position_std_m']),
                    rng.normal(0,cfg['initial_velocity_std_m_s']),
                    Rotation.from_rotvec(rng.normal(0,cfg['initial_attitude_std_rad'],3)).as_quat(),
                    rng.normal(0,cfg['initial_rate_std_rad_s'],3),
                    np.asarray(parameters['motor_gain'])*hover+parameters['motor_bias']]
    truth,states,commands = [initial.copy()],[initial[:13].copy()],[]
    prior = np.asarray(parameters['motor_gain'])*hover+parameters['motor_bias']
    poles = np.asarray(motor['poles']); gain,bias = np.asarray(motor['coefficients']).T
    previous = hover.copy(); hold=round(cfg['control_dt']/cfg['dt'])
    for k in range(round(cfg['warmup_s']/cfg['dt'])):
        if k%hold==0:
            ref,acc=at(0,'hover',cfg)
            u=bounded_command(pd_command(states[-1],ref,acc,parameters,cfg),previous,parameters,cfg)
            previous=u.copy()
        new=step(truth[-1],u,cfg['dt'],parameters,cfg['integration_dt'])
        states.append(observe(truth[-1],new,cfg['dt'])); truth.append(new)
        commands.append(u.copy()); prior=poles*prior+(1-poles)*(gain*u+bias)
    return {'truth':np.array(truth),'states':np.array(states),'commands':np.array(commands),
            'motor_prior':prior,'seed':seed}


def run_case(name,kind,seed,controller,initial,parameters,motor,cfg,case_dir,log,stage_started):
    states=list(initial['states'].copy()); commands=list(initial['commands'].copy())
    physical=initial['truth'][-1].copy(); prior=initial['motor_prior'].copy()
    previous=commands[-1].copy()
    if controller is not None: controller.reset()
    hold=round(cfg['control_dt']/cfg['dt']); nsteps=round(cfg['duration_s']/cfg['dt'])
    poles=np.asarray(motor['poles']); gain,bias=np.asarray(motor['coefficients']).T
    reference_cache=np.array([at(k*cfg['dt'],kind,cfg)[0] for k in range(nsteps+cfg['prediction_steps']+1)])
    physical_log=[physical.copy()]; measured_log=[states[-1].copy()]; inputs=[]; controls=[]
    reason='completed'; started=time.monotonic()
    for k in range(nsteps):
        if time.monotonic()-stage_started>cfg['stage_runtime_limit_s']:
            raise TimeoutError('Configured E3 stage runtime exceeded')
        if k%hold==0:
            control_start=time.perf_counter()
            ref,acc=at(k*cfg['dt'],kind,cfg)
            fallback=False
            if name=='pd':
                requested=pd_command(states[-1],ref,acc,parameters,cfg)
                stats={'success':True,'status':'pd','seconds':0.,'iterations':0,'objective':None}
            else:
                observation=pack_observation(states,commands,prior,controller.model,cfg)
                reference=reference_cache[k+np.arange(1,controller.blocks+1)*hold]
                requested,stats=controller.solve(observation,reference,previous)
                if not stats['success']:
                    fallback=True
                    requested=pd_command(states[-1],ref,acc,parameters,cfg)
            u=bounded_command(requested,previous,parameters,cfg)
            controls.append({'step':k,'t':k*cfg['dt'],**stats,'fallback':fallback,
                             'controller_wall_s':time.perf_counter()-control_start,
                             'bounded_adjustment':float(np.max(np.abs(u-requested))),
                             'max_command_change':float(np.max(np.abs(u-previous))),
                             **{f'u{j+1}':float(u[j]) for j in range(4)}})
            previous=u.copy()
            if k%200==0:
                log('case_progress',controller=name,trajectory=kind,seed=seed,simulated_s=k*cfg['dt'],
                    elapsed_case_s=round(time.monotonic()-started,2),fallbacks=sum(r['fallback'] for r in controls))
        new=step(physical,u,cfg['dt'],parameters,cfg['integration_dt'])
        finite=np.isfinite(new).all()
        if not finite:
            reason='nonfinite_plant'; break
        observation=observe(physical,new,cfg['dt'])
        states.append(observation); commands.append(u.copy())
        prior=poles*prior+(1-poles)*(gain*u+bias)
        physical=new
        physical_log.append(physical.copy()); measured_log.append(observation.copy()); inputs.append(u.copy())
        error=np.linalg.norm(physical[:3]-reference_cache[k+1,:3])
        tilt=np.degrees(np.arccos(np.clip(Rotation.from_quat(physical[6:10]).as_matrix()[2,2],-1,1)))
        if physical[2]<cfg['termination_min_height_m']: reason='minimum_height'
        elif error>cfg['termination_position_error_m']: reason='position_error'
        elif tilt>cfg['termination_tilt_deg']: reason='tilt'
        elif np.linalg.norm(physical[3:6])>cfg['termination_speed_m_s']: reason='speed'
        elif np.linalg.norm(physical[10:13])>cfg['termination_rate_rad_s']: reason='angular_rate'
        if reason!='completed': break
    complete=(len(inputs)==nsteps and reason=='completed')
    truth=np.asarray(physical_log); reference=reference_cache[:len(truth)]
    errors=np.linalg.norm(truth[:,:3]-reference[:,:3],axis=1)
    control_frame=pd.DataFrame(controls)
    applied=np.asarray(inputs).reshape(-1,4)
    violations=int(np.sum(np.any((applied<np.asarray(parameters['command_lower'])-1e-7)|(applied>np.asarray(parameters['command_upper'])+1e-7),axis=1)))
    slew_violations=int(np.sum(control_frame.max_command_change>cfg['command_delta_per_control']+1e-7))
    rmse=float(np.sqrt(np.mean(errors**2)))
    metrics={'controller':name,'trajectory':kind,'seed':seed,'complete':complete,'termination_reason':reason,
             'duration_completed_s':len(inputs)*cfg['dt'],'requested_duration_s':cfg['duration_s'],
             'position_rmse_m':rmse if complete else None,'prefix_position_rmse_m':rmse,
             'max_position_error_m':float(errors.max()),'final_position_error_m':float(errors[-1]),
             'tracking_success':bool(complete and rmse<=cfg['tracking_success_rmse_m'] and errors.max()<=cfg['tracking_success_max_error_m']
                                     and violations==0 and slew_violations==0),
             'solver_failures':int((~control_frame.success).sum()),'fallbacks':int(control_frame.fallback.sum()),
             'control_calls':len(controls),'input_bound_violations':violations,'input_slew_violations':slew_violations,
             'command_variation_squared':float(np.sum(np.diff(control_frame[[f'u{i}' for i in range(1,5)]].to_numpy(),axis=0)**2)),
             'command_squared_integral_code_units':float(np.sum(applied*applied)*cfg['dt']),
             'solve_time_median_s':float(control_frame.seconds.median()),'solve_time_p95_s':float(control_frame.seconds.quantile(.95)),
             'controller_time_p95_s':float(control_frame.controller_wall_s.quantile(.95)),
             'controller_over_period_fraction':float((control_frame.controller_wall_s>cfg['control_dt']).mean()),
             'case_wall_s':time.monotonic()-started}
    case_dir.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(case_dir/'trajectory.npz',t=np.arange(len(truth))*cfg['dt'],physical=truth,
                        observation=np.array(measured_log),reference=reference,command=applied,
                        initial_history_states=initial['states'],initial_history_commands=initial['commands'])
    control_frame.to_csv(case_dir/'control.csv',index=False)
    write_json(case_dir/'metrics.json',metrics)
    return metrics


def main(args):
    out=args.run_dir; start=time.monotonic()
    if args.stage in ['develop','freeze']:
        out.mkdir(parents=True,exist_ok=False)
        cfg=json.loads(args.config.read_text())
        if args.stage=='develop':
            if args.duration is not None: cfg['duration_s']=args.duration
            if args.trajectories: cfg['trajectories']=args.trajectories
            if args.controllers: cfg['controllers']=args.controllers
        calibrated=json.loads((args.calibration_dir/'CALIBRATED.json').read_text())
        assert calibrated['passed'],'Reference plant did not pass qualification'
        assert checksum(args.calibration_dir/'plant_parameters.json')==calibrated['plant_sha256']
        params=json.loads((args.calibration_dir/'plant_parameters.json').read_text())
        e2=ROOT/'runs/pelican_e2_v1'
        model_path=e2/'models/selected_models.json'
        freeze=json.loads((e2/'FROZEN.json').read_text())
        assert checksum(model_path)==freeze['selected_models_sha256']
        models_raw=json.loads(model_path.read_text()); motor=json.loads((e2/'motor_model.json').read_text())
        write_json(out/'config.json',cfg); write_json(out/'plant_parameters.json',params)
        write_json(out/'selected_models.json',models_raw); write_json(out/'motor_model.json',motor)
        write_json(out/'plant_qualification.json',calibrated)
        if args.stage=='freeze':
            development=ROOT/'runs/pelican_e3_dev_long_v2'
            development_rows=pd.read_csv(development/'tracking_per_run.csv')
            assert len(development_rows)==len(cfg['controllers'])
            assert bool(development_rows.complete.all()) and bool(development_rows.tracking_success.all())
            assert development_rows.input_bound_violations.sum()==development_rows.input_slew_violations.sum()==0
            dev_cfg=json.loads((development/'config.json').read_text())
            for key,value in cfg.items():
                if key!='trajectories': assert dev_cfg[key]==value,'Changed after long development: '+key
            write_json(out/'development_qualification.json',{'path':str(development.relative_to(ROOT)).replace('\\','/'),
                       'metrics_sha256':checksum(development/'tracking_per_run.csv'),'config_sha256':checksum(development/'config.json'),
                       'passed':True,'rule':'All five controllers complete 20 s circle with success thresholds and no input constraint violation.'})
            write_json(out/'environment.json',{'python':sys.version,'platform':platform.platform(),
                       'processor':platform.processor(),'logical_cpu_count':os.cpu_count(),'blas_threads':1,
                       'packages':{k:importlib.metadata.version(k) for k in ['numpy','scipy','casadi','pandas','matplotlib']}})
            sources=list((ROOT/'pelican_e3').glob('*.py'))+[Path(__file__),args.config,
                    ROOT/'pelican_e2/data.py',ROOT/'scripts/calibrate_pelican_e3.py',ROOT/'tests/test_pelican_e3.py']
            record={'frozen_utc':datetime.now(timezone.utc).isoformat(),'e2_selected_models_sha256':checksum(model_path),
                    'calibration_source':str(args.calibration_dir.relative_to(ROOT)).replace('\\','/'),
                    'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):checksum(p) for p in sources},
                    'artifact_sha256':{name:checksum(out/name) for name in ['config.json','plant_parameters.json','selected_models.json','motor_model.json','plant_qualification.json','development_qualification.json','environment.json']},
                    'scenarios':[{ 'trajectory':kind,'seed':seed} for kind in cfg['trajectories'] for seed in cfg['test_seeds']],
                    'evidence_type':'SIMULATED CLOSED LOOP','no_e2_test_data_used_for_calibration_or_tuning':True,
                    'plot_selection':'Fixed first test seed for each trajectory; show all controllers including failures.',
                    'primary_metric':'Full 20 s position vector RMSE; incomplete runs have null full-horizon RMSE and retain prefix metrics.'}
            write_json(out/'FROZEN.json',record); print(json.dumps(record,indent=2)); return
    else:
        assert not (out/'COMPLETED.json').exists(),'Formal test already completed'
        freeze=json.loads((out/'FROZEN.json').read_text())
        for path,expected in freeze['source_sha256'].items(): assert checksum(ROOT/path)==expected,path
        for path,expected in freeze['artifact_sha256'].items(): assert checksum(out/path)==expected,path
        cfg=json.loads((out/'config.json').read_text()); params=json.loads((out/'plant_parameters.json').read_text())
        models_raw=json.loads((out/'selected_models.json').read_text()); motor=json.loads((out/'motor_model.json').read_text())
    def log(event,**values):
        item={'utc':datetime.now(timezone.utc).isoformat(),'stage':args.stage,'elapsed_s':round(time.monotonic()-start,3),'event':event,**values}
        line=json.dumps(item)
        with (out/'progress.jsonl').open('a',encoding='utf-8') as f: f.write(line+'\n')
        print(line,flush=True)
    models={m['name']:restore_model(m) for m in models_raw}
    controllers={'pd':None}
    for name in cfg['controllers']:
        if name=='pd':continue
        began=time.monotonic(); controllers[name]=MPC(models[name],motor,params,cfg)
        log('controller_built',controller=name,seconds=round(time.monotonic()-began,3))
    seeds=cfg['development_seeds'] if args.stage=='develop' else cfg['test_seeds']
    initial={seed:initialize(seed,params,motor,cfg) for seed in seeds}
    (out/'initial_conditions').mkdir(exist_ok=True)
    for seed,value in initial.items(): np.savez_compressed(out/'initial_conditions'/f'{seed}.npz',**value)
    rows=[]
    for kind in cfg['trajectories']:
        for seed in seeds:
            for name in cfg['controllers']:
                log('case_started',controller=name,trajectory=kind,seed=seed)
                metrics=run_case(name,kind,seed,controllers[name],initial[seed],params,motor,cfg,
                                 out/'cases'/f'{kind}_{seed}_{name}',log,start)
                rows.append(metrics); pd.DataFrame(rows).to_csv(out/'tracking_per_run.csv',index=False)
                log('case_completed',**metrics)
    frame=pd.DataFrame(rows); summary=[]
    for (kind,name),group in frame.groupby(['trajectory','controller']):
        summary.append({'trajectory':kind,'controller':name,'runs':len(group),'completed':int(group.complete.sum()),
                        'tracking_successes':int(group.tracking_success.sum()),
                        'position_rmse_mean_successful_completion_only':float(group.position_rmse_m.mean()) if group.complete.any() else None,
                        'worst_position_error_m':float(group.max_position_error_m.max()),'fallbacks':int(group.fallbacks.sum()),
                        'solver_failures':int(group.solver_failures.sum()),'input_bound_violations':int(group.input_bound_violations.sum()),
                        'input_slew_violations':int(group.input_slew_violations.sum())})
    pd.DataFrame(summary).to_csv(out/'tracking_summary.csv',index=False)
    write_json(out/'COMPLETED.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'stage':args.stage,
                                   'cases':len(rows),'elapsed_s':time.monotonic()-start,'results':summary})
    log('all_cases_complete',cases=len(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['develop','freeze','test'],required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/pelican_e3.json')
    p.add_argument('--calibration-dir',type=Path,default=ROOT/'runs/pelican_e3_calibration_aero')
    p.add_argument('--duration',type=float)
    p.add_argument('--trajectories',nargs='+')
    p.add_argument('--controllers',nargs='+')
    with threadpool_limits(limits=1):main(p.parse_args())
