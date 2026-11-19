"""Audit E2 report numbers/plot data and record artifact hashes."""
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from PIL import Image
from run_pelican_e2 import checksum, write_json
from verify_pelican_e2 import verify_freeze


def main(run):
    _,splits = verify_freeze(run)
    config = json.loads((run/'config.json').read_text())
    report = json.loads((run/'analysis/report.json').read_text())
    frame = pd.read_csv(run/'test/prediction_per_flight.csv')
    primary = frame[frame.horizon_s==config['primary_horizon_s']]
    ids = sorted(splits['test'])
    with np.load(run/'analysis/bootstrap_draws.npz') as f:
        draws = f['draws']; assert_array_equal(f['flight_ids'],ids)
    expected = np.random.default_rng(config['bootstrap_seed']).integers(0,len(ids),size=draws.shape)
    assert_array_equal(draws,expected)
    pairs = pd.read_csv(run/'analysis/primary_paired_bootstrap.csv')
    for _,row in pairs.iterrows():
        metric = row.metric+'_rmse'
        base = primary[primary.model==row.baseline].set_index('flight_id').loc[ids,metric].to_numpy()
        new = primary[primary.model==row.candidate].set_index('flight_id').loc[ids,metric].to_numpy()
        deltas = [sum(float(base[i]-new[i]) for i in draw)/len(draw) for draw in draws]
        relative = [100*d/(sum(float(base[i]) for i in draw)/len(draw)) for d,draw in zip(deltas,draws)]
        assert_allclose([row.lower_95,row.upper_95],np.percentile(deltas,[2.5,97.5]),atol=1e-12)
        assert_allclose([row.percent_lower_95,row.percent_upper_95],np.percentile(relative,[2.5,97.5]),atol=1e-11)
        assert_allclose(row.improvement_absolute,np.mean(base-new),atol=1e-12)
        assert_allclose(row.improvement_percent,100*(np.mean(base)-np.mean(new))/np.mean(base),atol=1e-11)
        assert row.flights_improved == sum(base>new)
    assert_allclose(pairs.select_dtypes('number'),pd.DataFrame(report['primary_paired_bootstrap']).select_dtypes('number'),atol=1e-12)
    summary = pd.read_csv(run/'test/prediction_summary.csv')
    for row in report['primary_results']:
        saved = summary[(summary.model==row['model'])&(summary.horizon_s==row['horizon_s'])].iloc[0]
        for metric in ['velocity','omega','position','attitude']:
            assert_allclose(row[metric+'_rmse_mean'],saved[metric+'_rmse_mean'],atol=1e-12)
    baseline = primary[primary.model=='sindy_current'].sort_values(['velocity_rmse','flight_id'])
    expected_ids = [int(baseline.iloc[len(baseline)//2].flight_id),int(baseline.iloc[-1].flight_id)]
    examples = report['representative_examples']
    assert [e['flight_id'] for e in examples] == expected_ids
    for e in examples:
        fid,origin = e['flight_id'],e['origin_index']
        with np.load(run/'analysis'/f'example_{e["designation"]}_flight_{fid:02d}.npz') as example:
            for name in [k for k in example.files if k not in ['time','truth']]:
                with np.load(run/'test'/f'{name}_endpoints.npz') as endpoints:
                    for horizon in config['horizons_s']:
                        step = round(horizon/config['dt']); prefix = f'f{fid:02d}_s{step:03d}'
                        index = int(np.flatnonzero(endpoints[prefix+'_origins']==origin)[0])
                        assert_allclose(example[name][step],endpoints[prefix+'_prediction'][index],atol=2e-8,rtol=2e-8,equal_nan=True)
                        assert_array_equal(example['truth'][step],endpoints[prefix+'_truth'][index])
    image_meta = []
    for p in sorted((run/'figures').glob('*.png')):
        with Image.open(p) as im:
            assert im.width>=2000 and im.info['dpi'][0]>=299
            image_meta.append({'file':p.name,'width':im.width,'height':im.height,'dpi':im.info['dpi']})
        if not p.stem.endswith('_grayscale'):
            assert p.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
    assert len(image_meta)==10
    # Validate the links in the new human-readable report.
    import re
    doc = ROOT/'docs/PELICAN_E2_RESULTS.md'
    for href in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        if not href.startswith(('http:', 'https:', '#')): assert (doc.parent/href).exists(),href
    result = {'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
              'paired_comparisons_recomputed':len(pairs),'example_endpoints_match_test':True,
              'report_links_resolve':True,'image_metadata':image_meta,
              'visual_qa':'Color figures inspected; horizon and example grayscale inspected. Overlapping flight labels replaced with paired rows; failed example marked at 0.66 s.',
              'verifier_sha256':checksum(Path(__file__))}
    write_json(run/'analysis_verification.json',result)
    code = [ROOT/'scripts'/name for name in ['run_pelican_e2.py','verify_pelican_e2.py','report_pelican_e2.py','verify_pelican_e2_report.py']]
    code += list((ROOT/'pelican_e2').glob('*.py'))+[ROOT/'configs/pelican_e2.json',ROOT/'tests/test_pelican_e2.py']
    docs = [doc,ROOT/'README.md',ROOT/'docs/PELICAN_EXPERIMENT_PLAN.md',ROOT/'docs/IMPLEMENTATION_STATUS.md']
    files = [p for p in run.rglob('*') if p.is_file() and p.name!='artifact_hashes.json']+code+docs
    manifest = {str(p.relative_to(ROOT)).replace('\\','/'):{'sha256':checksum(p),'bytes':p.stat().st_size} for p in sorted(set(files))}
    write_json(run/'artifact_hashes.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'files':manifest})
    print(json.dumps({'status':'PASS','paired_comparisons':len(pairs),'checked_images':len(image_meta),'hashed_artifacts':len(manifest)},indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/pelican_e2_v1')
    main(parser.parse_args().run_dir)
