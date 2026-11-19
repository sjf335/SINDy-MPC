"""Audit post-hoc diagnostic report summaries and seal the separate artifact set."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from PIL import Image
from run_pelican_e2 import checksum,write_json

OUT=ROOT/'runs/pelican_e3_diagnostic_v1'


def main():
    verification=json.loads((OUT/'verification.json').read_text());assert verification['status']=='PASS'
    cases=pd.read_csv(OUT/'all_intervention_cases.csv')
    original=pd.concat([pd.read_csv(OUT/'intervention/per_run.csv'),pd.read_csv(OUT/'bias_components/per_run.csv')],ignore_index=True)
    pd.testing.assert_frame_equal(cases,original)
    summary=pd.read_csv(OUT/'intervention_summary.csv')
    for row in summary.itertuples(index=False):
        f=cases[(cases.controller==row.controller)&(cases.variant==row.variant)]
        assert row.cases==len(f) and row.completed==int(f.complete.sum())
        assert row.tracking_successes==int(f.tracking_success.sum()) and row.fallbacks==int(f.fallbacks.sum())
        assert_allclose([row.mean_rmse_m,row.min_rmse_m,row.max_rmse_m],
                        [f.position_rmse_m.mean(),f.position_rmse_m.min(),f.position_rmse_m.max()],atol=1e-12,equal_nan=True)
    paired=pd.read_csv(OUT/'paired_intervention_differences.csv')
    for row in paired.itertuples(index=False):
        g=cases[cases.controller==row.controller].pivot(index=['trajectory','seed'],columns='variant',values='position_rmse_m')
        a,b=g.original.to_numpy(),g.hover_corrected.to_numpy()
        assert row.pairs==len(a) and row.improved_cases==int(np.sum(a>b))
        assert_allclose([row.original_m,row.corrected_m,row.difference_m,row.relative_reduction_percent],
                        [a.mean(),b.mean(),np.mean(a-b),100*np.mean(a-b)/a.mean()],atol=1e-10)
    late=pd.read_csv(OUT/'intervention_late_errors.csv')
    for row in late.itertuples(index=False):
        folder='intervention' if row.variant in ['original','hover_corrected'] else 'bias_components'
        with np.load(OUT/folder/'cases'/f'{row.trajectory}_{row.seed}_{row.controller}_{row.variant}'/'trajectory.npz') as f:
            e=f['physical'][f['t']>=10,:3]-f['reference'][f['t']>=10,:3]
        assert_allclose([row.late_rmse_m,row.late_bias_norm_m],
                        [np.sqrt(np.mean(np.sum(e*e,axis=1))),np.linalg.norm(e.mean(0))],atol=1e-12)
    raw_sensitivity=pd.read_csv(OUT/'input_sensitivity.csv')
    sensitivity_summary=pd.read_csv(OUT/'input_sensitivity_summary.csv')
    for row in sensitivity_summary.itertuples(index=False):
        g=raw_sensitivity[(raw_sensitivity.model==row.model)&(raw_sensitivity.horizon_s==row.horizon_s)&
                          (raw_sensitivity.amplitude==row.amplitude)]
        g=g[g.probe=='hover'] if row.region=='hover' else g[g.probe!='hover']
        for field in ['p','v','w']:
            a=g[[f'plant_{field}_{axis}' for axis in 'xyz']].to_numpy()
            b=g[[f'model_{field}_{axis}' for axis in 'xyz']].to_numpy()
            good=np.isfinite(b).all(1)
            assert getattr(row,f'{field}_invalid_rows')==int((~good).sum())
            a,b=a[good],b[good]
            norm_a=np.sqrt(np.sum(a*a));norm_b=np.sqrt(np.sum(b*b))
            expected=[np.sqrt(np.sum((b-a)**2))/norm_a,norm_b/norm_a,np.sum(a*b)/(norm_a*norm_b)]
            assert_allclose([getattr(row,f'{field}_{suffix}') for suffix in ['relative_error','gain_ratio','cosine']],expected,atol=1e-10)
    docs=[ROOT/'docs/PELICAN_E3_DIAGNOSTICS.md',ROOT/'docs/PELICAN_E3_DIAGNOSTIC_PROTOCOL.md']
    for doc in docs:
        for href in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if not href.startswith(('http:','https:','#')):assert (doc.parent/href).exists(),href
    figures=[]
    for path in sorted((OUT/'figures').glob('*.png')):
        with Image.open(path) as im:
            assert im.width>=2000 and im.info['dpi'][0]>=299
            figures.append(dict(name=path.name,size=list(im.size),dpi=list(im.info['dpi'])))
        if not path.stem.endswith('_grayscale'):assert path.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
    assert len(figures)==6
    # Original E3 report and run remain byte-for-byte unchanged.
    prior=json.loads((ROOT/'runs/pelican_e3_v1/artifact_hashes.json').read_text())
    for path,record in prior['files'].items():assert checksum(ROOT/path)==record['sha256']
    result=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),cases=len(cases),summary_rows=len(summary),
                paired_rows=len(paired),late_error_rows=len(late),sensitivity_summary_rows=len(sensitivity_summary),
                figures=figures,old_e3_files_unchanged=len(prior['files']),
                verifier_sha256=checksum(Path(__file__)))
    write_json(OUT/'analysis_verification.json',result)
    scripts=[ROOT/'scripts'/name for name in ['diagnose_pelican_e3.py','diagnose_pelican_e3_bias_components.py',
             'verify_pelican_e3_diagnostics.py','report_pelican_e3_diagnostics.py','verify_pelican_e3_diagnostic_report.py']]
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json']+docs+scripts
    write_json(OUT/'artifact_hashes.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
          files={str(p.relative_to(ROOT)).replace('\\','/'):dict(sha256=checksum(p),bytes=p.stat().st_size) for p in sorted(set(paths))}))
    print(json.dumps({k:v for k,v in result.items() if k!='figures'},indent=2))


if __name__=='__main__':main()
