"""Recompute report intervals/timing and hash E3 deliverables."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose,assert_array_equal
from PIL import Image
from run_pelican_e2 import checksum,write_json


def main(run):
    cfg=json.loads((run/'config.json').read_text())
    freeze=json.loads((run/'FROZEN.json').read_text())
    for name,expected in freeze['source_sha256'].items(): assert checksum(ROOT/name)==expected
    for name,expected in freeze['artifact_sha256'].items(): assert checksum(run/name)==expected
    report=json.loads((run/'analysis/report.json').read_text())
    original=pd.read_csv(run/'tracking_per_run.csv')
    stats=pd.read_csv(run/'analysis/tracking_with_intervals.csv')
    pairs=pd.read_csv(run/'analysis/paired_comparisons.csv')
    with np.load(run/'analysis/bootstrap_draws.npz') as f:
        seeds=f['seeds'];draws=f['draws']
    assert_array_equal(seeds,cfg['test_seeds'])
    assert_array_equal(draws,np.random.default_rng(cfg['bootstrap_seed']).integers(0,len(seeds),size=draws.shape))
    for row in stats.itertuples(index=False):
        f=original[original.controller==row.controller]
        if row.trajectory!='overall':f=f[f.trajectory==row.trajectory]
        assert row.runs==len(f) and row.completed==int(f.complete.sum())
        assert row.tracking_successes==int(f.tracking_success.sum())
        assert_allclose(row.position_rmse_mean_m,f.position_rmse_m.mean(),atol=1e-12,equal_nan=True)
        if f.complete.all():
            values=np.array([f[f.seed==seed].position_rmse_m.mean() for seed in seeds])
            sampled=np.array([sum(float(values[j]) for j in draw)/len(draw) for draw in draws])
            assert_allclose([row.lower_95_m,row.upper_95_m],np.percentile(sampled,[2.5,97.5]),atol=1e-12)
        else: assert row.conditional_on_completion and np.isnan(row.lower_95_m)
    for row in pairs.itertuples(index=False):
        f=original if row.trajectory=='overall' else original[original.trajectory==row.trajectory]
        pivot=f.pivot(index=['trajectory','seed'],columns='controller',values='position_rmse_m')
        common=pivot[[row.baseline,row.candidate]].dropna()
        assert row.available_pairs==len(common)
        difference=common[row.baseline]-common[row.candidate]
        assert_allclose(row.difference_baseline_minus_history_m,difference.mean(),atol=1e-12,equal_nan=True)
        assert row.history_better_cases==int((difference>0).sum())
        if row.all_pairs_complete:
            byseed=difference.groupby('seed').mean().reindex(seeds).to_numpy()
            replicate=[sum(float(byseed[j]) for j in draw)/len(draw) for draw in draws]
            assert_allclose([row.lower_95_m,row.upper_95_m],np.percentile(replicate,[2.5,97.5]),atol=1e-12)
    all_controls=pd.read_csv(run/'analysis/all_control_calls.csv')
    assert len(all_controls)==int(original.control_calls.sum())
    timing=pd.read_csv(run/'analysis/timing_summary.csv')
    for row in timing.itertuples(index=False):
        values=all_controls[all_controls.controller==row.controller][row.measurement].to_numpy()
        assert row.calls==len(values)
        assert_allclose([row.mean_s,row.median_s,row.p95_s,row.p99_s,row.max_s],
                        [np.mean(values),np.median(values),np.percentile(values,95),np.percentile(values,99),np.max(values)],atol=1e-12)
        assert_allclose(row.fraction_over_period,np.mean(values>cfg['control_dt']),atol=1e-12)
    assert report['cases']==len(original) and report['completed']==int(original.complete.sum())
    assert report['fallbacks']==int(original.fallbacks.sum())
    images=[]
    for file in sorted((run/'figures').glob('*.png')):
        with Image.open(file) as im:
            assert im.width>=2000 and im.info['dpi'][0]>=299
            images.append({'file':file.name,'size':list(im.size),'dpi':list(im.info['dpi'])})
        if not file.stem.endswith('_grayscale'): assert file.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
    assert len(images)>=8
    docs=[ROOT/'docs/PELICAN_E3_PROTOCOL.md',ROOT/'docs/PELICAN_E3_RESULTS.md']
    for doc in docs:
        assert doc.exists()
        for href in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if not href.startswith(('http:','https:','#')): assert (doc.parent/href).exists(),href
    result={'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),'summary_groups':len(stats),
            'paired_comparisons':len(pairs),'timing_groups':len(timing),'control_calls':len(all_controls),
            'images':images,'report_links_resolve':True,'frozen_source_and_models_unchanged':True,
            'verifier_sha256':checksum(Path(__file__))}
    write_json(run/'analysis_verification.json',result)
    scripts=[ROOT/'scripts'/name for name in ['verify_pelican_e3.py','report_pelican_e3.py','verify_pelican_e3_report.py']]
    files=[p for p in run.rglob('*') if p.is_file() and p.name!='artifact_hashes.json']
    files += [ROOT/path for path in freeze['source_sha256']]+docs+scripts
    files += [ROOT/'README.md',ROOT/'docs/PELICAN_EXPERIMENT_PLAN.md',ROOT/'docs/IMPLEMENTATION_STATUS.md']
    write_json(run/'artifact_hashes.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
         'files':{str(p.relative_to(ROOT)).replace('\\','/'):{'sha256':checksum(p),'bytes':p.stat().st_size} for p in sorted(set(files))}})
    print(json.dumps({k:v for k,v in result.items() if k!='images'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/pelican_e3_v1')
    main(parser.parse_args().run_dir)
