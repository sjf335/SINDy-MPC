"""Adaptive sanity check: is the same causal filter alone destabilizing history?

Added after observing some noisy-filter failures. No tuning or replacement of
the primary experiment. Same two seeds/two paths, two historical controllers.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_pelican_e2 import restore_model,checksum,write_json
from run_pelican_e3_nominal_comparison import log
from run_pelican_e4_estimation_diagnostic import run_case,preserved_e4
from pelican_e3_corrected.control import BiasMPC


def main(args):
    parent=args.parent.resolve();out=args.run_dir.resolve()
    design=json.loads((parent/'design.json').read_text())
    design.update(no_noise_filter_check=True,controllers=['linear_history','sindy_history'],
                  primary_modes=['causal_poly'],observer_ablation_controllers=[],observer_ablation_modes=[],
                  notes=['Adaptive supplementary sanity check added after observing noisy-filter history failures.',
                         'Same fixed filter and controller parameters; only injected pose noise is removed.',
                         '8 new cases, no retuning. Not independent confirmatory evidence.'])
    source=ROOT/design['base_run'];preserved=preserved_e4(source)
    settings=json.loads((parent/'config.json').read_text());cfg=json.loads((parent/'e3_config.json').read_text())
    params=json.loads((parent/'plant_parameters.json').read_text());motor=json.loads((parent/'motor_model.json').read_text())
    models={k:restore_model(v) for k,v in json.loads((parent/'corrected_models.json').read_text()).items()}
    plan=[dict(mode='causal_poly',bias_enabled=True,controller=n,trajectory=t,seed=s,
               case_id=f'causal_poly_bias_on_{t}_{s}_{n}')
          for s in design['seeds'] for t in design['trajectories'] for n in design['controllers']]
    out.mkdir(parents=True,exist_ok=False);(out/'initial_conditions').mkdir()
    for file in ['plant_parameters.json','motor_model.json','corrected_models.json','equilibrium_calibration.json','e3_config.json']:
        (out/file).write_bytes((parent/file).read_bytes())
    initial={}
    for seed in design['seeds']:
        p=parent/'initial_conditions'/f'{seed}.npz';(out/'initial_conditions'/p.name).write_bytes(p.read_bytes())
        with np.load(p) as a:initial[seed]={k:a[k] for k in a.files}
    write_json(out/'design.json',design);write_json(out/'config.json',settings);write_json(out/'planned_cases.json',plan)
    hashes=json.loads((parent/'FROZEN.json').read_text())['source_sha256']
    hashes[Path(__file__).relative_to(ROOT).as_posix()]=checksum(Path(__file__))
    write_json(out/'FROZEN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        stage='adaptive_filter_sanity',parent=parent.relative_to(ROOT).as_posix(),source_sha256=hashes,
        base_manifest_sha256=checksum(source/'artifact_hashes.json'),preserved_files=preserved,
        artifact_sha256={p.relative_to(out).as_posix():checksum(p) for p in out.rglob('*') if p.is_file()}))
    controllers={n:BiasMPC(models[n],motor,params,cfg) for n in design['controllers']}
    scenario=next(s for s in settings['scenarios'] if s['name']=='nominal')
    started=time.monotonic();rows=[]
    for c in plan:
        r=run_case(scenario,c['controller'],c['trajectory'],c['seed'],controllers[c['controller']],
            initial[c['seed']],params,motor,cfg,settings,out/'cases'/c['case_id'],
            lambda event,**v:log(out,event,case_id=c['case_id'],**v),started,
            mode='causal_poly',bias_enabled=True,window=design['filter_window_samples'])
        r['case_id']=c['case_id'];rows.append(r);pd.DataFrame(rows).to_csv(out/'diagnostic_per_run.csv',index=False)
        log(out,'case_complete',done=len(rows),total=len(plan),**r)
    for p,h in hashes.items():assert checksum(ROOT/p)==h,p
    preserved_e4(source)
    write_json(out/'COMPLETED.json',dict(cases=len(rows),elapsed_s=time.monotonic()-started,
        complete=sum(r['complete'] for r in rows),tracking_sha256=checksum(out/'diagnostic_per_run.csv')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    with threadpool_limits(limits=1):main(p.parse_args())
