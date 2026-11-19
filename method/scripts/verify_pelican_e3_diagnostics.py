"""Independent diagnostic metric, finite-difference and paired-intervention audit."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose,assert_array_equal
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from diagnose_pelican_e3 import SOURCE,load,episode
from pelican_e3.control import symbolic_transition,pack_observation
from run_pelican_e2 import checksum,write_json
from verify_pelican_e3 import reference_derivative


def verify(out,offline):
    cfg,p,motor,models=load();report={}
    started=json.loads((out/'STARTED.json').read_text())
    assert checksum(ROOT/'scripts/diagnose_pelican_e3.py')==started['script_sha256']
    assert checksum(ROOT/'docs/PELICAN_E3_DIAGNOSTIC_PROTOCOL.md')==started['protocol_sha256']
    # Preserve all previously sealed E3 files, including report artifacts.
    old_manifest=json.loads((SOURCE/'artifact_hashes.json').read_text())
    for name,record in old_manifest['files'].items():assert checksum(ROOT/name)==record['sha256'],name
    report['e3_sealed_files_unchanged']=len(old_manifest['files'])
    table=pd.read_csv(out/'replay_per_window.csv');index=pd.read_csv(out/'replay_endpoint_index.csv')
    assert len(table)==len(index)==75*37*4*6
    with np.load(out/'replay_endpoints.npz') as f:pred=f['predicted'];truth=f['truth']
    for start,label in [(0,'position_sq_m2'),(3,'velocity_sq_m2_s2'),(10,'omega_sq_rad2_s2')]:
        assert_allclose(np.sum((pred[:,start:start+3]-truth[:,start:start+3])**2,axis=1),table[label],atol=1e-12,equal_nan=True)
    good=np.isfinite(pred).all(1);assert_array_equal(~good,table.failed)
    angle=(Rotation.from_quat(truth[good,6:10]).inv()*Rotation.from_quat(pred[good,6:10])).magnitude()
    assert_allclose(angle**2,table.loc[good,'attitude_sq_rad2'],atol=1e-12)
    report['replay_endpoints_recomputed']=len(table)
    summary=pd.read_csv(out/'replay_summary.csv')
    for row in summary.itertuples(index=False):
        g=table[(table.model==row.model)&(table.horizon_s==row.horizon_s)]
        if row.source_controller!='all':g=g[g.source_controller==row.source_controller]
        assert row.windows==len(g) and row.failures==g.failed.sum()
        for field,attr in [('position_sq_m2','position_rmse_m'),('velocity_sq_m2_s2','velocity_rmse_m_s'),
                           ('omega_sq_rad2_s2','omega_rmse_rad_s'),('attitude_sq_rad2','attitude_rmse_rad')]:
            values=[np.sqrt(part[field].mean()) for _,part in g.groupby(['trajectory','seed','source_controller'])]
            assert_allclose(getattr(row,attr),np.mean(values),atol=1e-12)
    # Independent symbolic route for selected saved replay windows.
    model_error=0.;windows=0
    for kind in ['circle','figure8','helix']:
        for source in ['pd','sindy_history','motor_sindy']:
            e,t,n=episode(SOURCE/'cases'/f'{kind}_8301_{source}',motor)
            for name,model in models.items():
                fn,_=symbolic_transition(model,motor,cfg)
                for k in [200,1000]:
                    origin=n+k
                    packed=pack_observation(e['x'][:origin+1],e['u'][:origin],e['motor_hat'][origin-1],model,cfg)
                    for j in range(50):packed=np.asarray(fn(packed,e['u'][origin+j])).ravel()
                    mask=((index.trajectory==kind)&(index.seed==8301)&(index.source_controller==source)&
                          (index.model==name)&(index.origin_step==k)&(index.horizon_s==.5))
                    assert mask.sum()==1
                    stored=pred[np.flatnonzero(mask)[0]]
                    model_error=max(model_error,float(np.max(np.abs(packed[:13]-stored))))
                    assert_allclose(packed[:13],stored,atol=1e-9,rtol=1e-9);windows+=1
    report.update(symbolic_windows_checked=windows,max_symbolic_difference=model_error)
    decomp=pd.read_csv(out/'late_error_decomposition.csv')
    for row in decomp.itertuples(index=False):
        with np.load(SOURCE/'cases'/f'{row.trajectory}_{row.seed}_{row.controller}'/'trajectory.npz') as f:
            error=f['physical'][f['t']>=10,:3]-f['reference'][f['t']>=10,:3]
        bias=np.mean(error,axis=0);total=np.mean(np.einsum('ij,ij->i',error,error));bias2=bias@bias
        assert_allclose([row.late_rmse_m,row.late_bias_norm_m,row.bias_fraction_mse],
                        [np.sqrt(total),np.sqrt(bias2),bias2/total],atol=1e-12)
        assert_allclose(row.late_demeaned_rmse_m**2,total-bias2,atol=1e-12)
    # Every tabulated central difference is recomputed from saved branch endpoints.
    data=np.load(out/'input_response_endpoints.npz');meta=data['metadata'];labels=json.loads((out/'input_probe_labels.json').read_text())
    sensitivity=pd.read_csv(out/'input_sensitivity.csv');count=0
    lookup={(labels[int(m[0])],float(m[1]),int(m[2])+1):j for j,m in enumerate(meta) if int(m[3])==1}
    for row in sensitivity.itertuples(index=False):
        j=lookup[(row.probe,row.amplitude,row.rotor)];k=round(row.horizon_s/cfg['dt'])
        for prefix,raw in [('plant',data[f'plant_{k}']),('model',data[f'{row.model}_{k}'])]:
            derivative=(raw[j]-raw[j+1])/(2*row.amplitude)
            for group,start in [('p',0),('v',3),('w',10)]:
                assert_allclose([getattr(row,f'{prefix}_{group}_{a}') for a in 'xyz'],derivative[start:start+3],atol=1e-12)
        count+=1
    report['finite_difference_rows_recomputed']=count
    a=sensitivity[sensitivity.amplitude==.25].sort_values(['model','probe','rotor','horizon_s'])
    b=sensitivity[sensitivity.amplitude==1.].sort_values(['model','probe','rotor','horizon_s'])
    fd_difference={}
    for column in [c for c in sensitivity if c.startswith(('plant_','model_'))]:
        va,vb=a[column].to_numpy(),b[column].to_numpy()
        fd_difference[column]=float(np.linalg.norm(va-vb)/max(1e-12,np.linalg.norm(va)))
    report['max_amplitude_relative_difference']=max(fd_difference.values())
    assert report['max_amplitude_relative_difference']<.01
    # Separate continuous equations and adaptive integrator check long branch responses.
    physical_error=0.;physical_checks=0
    for probe_id in [0,1,7,18,36,60,75]:
        label=labels[probe_id]
        if probe_id==0:
            u=np.array(p['hover_command'])
            state=np.r_[0.,0.,1.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,np.array(p['motor_gain'])*u+p['motor_bias']]
        else:
            case,second=label.rsplit('_t',1)
            with np.load(SOURCE/'cases'/case/'trajectory.npz') as f:state=f['physical'][round(float(second)/cfg['dt'])]
        ids=[j for j,m in enumerate(meta) if int(m[0])==probe_id and float(m[1])==.25]
        for j in ids:
            command=data['commands'][j]
            sol=solve_ivp(lambda t,s:reference_derivative(s,command,p),[0,.5],state,rtol=1e-11,atol=1e-12,dense_output=True)
            for k in [1,5,10,25,50]:
                before,after=sol.sol([(k-1)*cfg['dt'],k*cfg['dt']]).T
                observed=after[:13].copy();observed[3:6]=(after[:3]-before[:3])/cfg['dt']
                observed[10:13]=(Rotation.from_quat(before[6:10]).inv()*Rotation.from_quat(after[6:10])).as_rotvec()/cfg['dt']
                delta=np.max(np.abs(observed-data[f'plant_{k}'][j]));physical_error=max(physical_error,float(delta))
                assert_allclose(observed,data[f'plant_{k}'][j],atol=3e-6,rtol=2e-5)
            physical_checks+=1
    data.close();report.update(adaptive_branch_trajectories=physical_checks,max_adaptive_observation_difference=physical_error)
    if not offline:
        records=[];transitions=0;initials={};comparison=[]
        for folder in ['intervention','bias_components']:
            complete=json.loads((out/folder/'COMPLETED.json').read_text())
            f=pd.read_csv(out/folder/'per_run.csv');assert len(f)==complete['cases']
            for row in f.itertuples(index=False):
                path=out/folder/'cases'/f'{row.trajectory}_{row.seed}_{row.controller}_{row.variant}'
                with np.load(path/'trajectory.npz') as raw:t={k:raw[k] for k in raw.files}
                error=np.linalg.norm(t['physical'][:,:3]-t['reference'][:,:3],axis=1)
                rmse=np.sqrt(np.mean(error**2));is_complete=len(t['command'])==2000 and row.termination_reason=='completed'
                assert row.complete==is_complete
                assert_allclose(row.prefix_position_rmse_m,rmse,atol=1e-12)
                if row.complete:assert_allclose(row.position_rmse_m,rmse,atol=1e-12)
                else:assert np.isnan(row.position_rmse_m)
                key=(row.trajectory,row.seed)
                if key in initials:
                    for name in ['initial_history_states','initial_history_commands']:assert_array_equal(t[name],initials[key][name])
                    assert_array_equal(t['physical'][0],initials[key]['physical'][0])
                else:initials[key]=t
                controls=pd.read_csv(path/'control.csv')
                assert row.fallbacks==int(controls.fallback.sum()) and row.solver_failures==int((~controls.success).sum())
                actual=t['command'];assert np.all(actual>=np.array(p['command_lower'])-1e-7) and np.all(actual<=np.array(p['command_upper'])+1e-7)
                held=np.repeat(controls[[f'u{i}' for i in range(1,5)]].to_numpy(),5,axis=0)[:len(actual)]
                assert_allclose(actual,held,atol=1e-12)
                changes=np.diff(np.vstack([t['initial_history_commands'][-1],controls[[f'u{i}' for i in range(1,5)]].to_numpy()]),axis=0)
                assert np.max(np.abs(changes))<=cfg['command_delta_per_control']+1e-7
                for k in [0,min(100,len(actual)-1),len(actual)-1]:
                    solution=solve_ivp(lambda time,s:reference_derivative(s,actual[k],p),[0,cfg['dt']],t['physical'][k],rtol=1e-11,atol=1e-12).y[:,-1]
                    assert_allclose(solution,t['physical'][k+1],atol=3e-7,rtol=3e-7);transitions+=1
                records.append(dict(controller=row.controller,trajectory=row.trajectory,seed=row.seed,variant=row.variant,
                                    complete=row.complete,rmse_m=row.position_rmse_m))
        report.update(intervention_cases_checked=len(records),intervention_adaptive_transitions=transitions,paired_conditions=len(initials))
        freeze=json.loads((out/'intervention/FROZEN.json').read_text())
        assert checksum(ROOT/'scripts/diagnose_pelican_e3.py')==freeze['script_sha256']
        adaptive=json.loads((out/'bias_components/ADAPTIVE_PROTOCOL.json').read_text())
        assert checksum(ROOT/'scripts/diagnose_pelican_e3_bias_components.py')==adaptive['source_script_sha256']
    report.update(status='PASS',utc=datetime.now(timezone.utc).isoformat(),scope='offline' if offline else 'all',
                  verifier_sha256=checksum(Path(__file__)),limitations='Same-assistant independent code-path audit; no real-plant fidelity guarantee.')
    write_json(out/('offline_verification.json' if offline else 'verification.json'),report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/pelican_e3_diagnostic_v1')
    parser.add_argument('--offline',action='store_true')
    a=parser.parse_args()
    with threadpool_limits(limits=1):verify(a.run_dir,a.offline)
