"""Recompute E1 validation metrics with direct recurrences/convolution.

Does not import the experiment's prediction or evaluation implementation.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/pelican_e1_v1')
    args = parser.parse_args()
    out = args.run_dir
    config = json.loads((out/'config.json').read_text())
    models = json.loads((out/'motor_response_models.json').read_text())
    splits = json.loads((out/'splits.json').read_text())
    manifest = json.loads((out/'code_manifest.json').read_text())
    for rel, checksum in manifest.items():
        assert digest(ROOT/rel) == checksum, f'Experiment source changed: {rel}'
    provenance = json.loads((out/'raw_provenance.json').read_text())
    raw = ROOT/'data/pelican/raw/AscTec_Pelican_Flight_Dataset.mat'
    assert digest(raw) == provenance['sha256']
    sets = [set(splits[k]) for k in ['train', 'validation', 'test']]
    assert len(set.union(*sets)) == 54 and sum(map(len, sets)) == 54
    allmetrics = pd.read_csv(out/'validation_all_candidates.csv')
    assert set(allmetrics.flight_id) == sets[1]
    assert set(pd.read_csv(out/'motor_diagnostics.csv').flight_id) == sets[0]
    metrics = pd.read_csv(out/'validation_per_flight_motor.csv').set_index(['model', 'flight_id', 'motor'])
    flights = loadmat(raw, simplify_cells=True)['flights']
    start, warm = config['prediction_start'], config['warmup_samples']
    scorestart = start+warm
    largest_error = 0.0
    nchecks = 0
    representative = pd.read_csv(out/'representative_response.csv')
    info = json.loads((out/'report.json').read_text())['figures']
    plot_indices = np.rint(representative['t_s'].to_numpy()/config['dt']).astype(int)
    for fid in splits['validation']:
        flight = flights[fid-1]
        u, y = flight['Motors_CMD'], flight['Motors']
        for model in models:
            coefficients = np.asarray(model['coefficients'])
            prediction = np.full_like(y, np.nan)
            if model['kind'] == 'first_order':
                poles = np.asarray(model['poles'])
                prev = y[start-1].copy()
                for k in range(start, len(u)):
                    prev = poles*prev+(1-poles)*(coefficients[:, 0]*u[k-model['delay']]+coefficients[:, 1])
                    prediction[k] = prev
            else:
                z = (u-np.asarray(model['mu']))/np.asarray(model['scale'])
                for m in range(4):
                    if model['kind'] == 'static':
                        prediction[:, m] = np.polynomial.polynomial.polyval(z[:, m], coefficients[m])
                    else:
                        prediction[:, m] = coefficients[m, 0]+np.convolve(z[:, m], coefficients[m, 1:])[:len(z)]
            for m in range(4):
                rmse = float(np.sqrt(np.mean((prediction[scorestart:, m]-y[scorestart:, m])**2)))
                saved = metrics.loc[(model['name'], fid, m+1), 'rmse_code_units']
                largest_error = max(largest_error, abs(rmse-saved))
                assert abs(rmse-saved) < 1e-9
                nchecks += 1
                if fid == info['representative_flight']:
                    np.testing.assert_allclose(prediction[plot_indices, m],
                                               representative[f'm{m+1}_{model["name"]}'], atol=1e-9)
    summary = pd.read_csv(out/'validation_summary.csv').set_index('model')
    grouped = metrics.reset_index().groupby('model').rmse_code_units.mean()
    for name, score in grouped.items():
        assert abs(score-summary.loc[name, 'mean_flight_motor_rmse']) < 1e-10
    selected = {m['kind']: m for m in models}
    candidates = pd.read_csv(out/'validation_model_selection.csv')
    for kind in ['fir', 'first_order']:
        subset = candidates[candidates.kind == kind]
        assert selected[kind]['name'] == subset.loc[subset.validation_score.idxmin(), 'model']
    f = metrics.reset_index().groupby(['flight_id', 'model']).rmse_code_units.mean().unstack()
    first, fir = selected['first_order']['name'], selected['fir']['name']
    diff = (f[first]-f[fir]).to_numpy()
    rng = np.random.default_rng(config['bootstrap_seed'])
    idx = rng.integers(0, len(diff), size=(config['bootstrap_replicates'], len(diff)))
    ci = np.quantile(diff[idx].mean(axis=1), [.025, .975])
    result = {'status': 'passed', 'validation_metric_rows_recomputed': nchecks,
              'maximum_rmse_difference': largest_error,
              'raw_sha256_verified': True, 'experiment_source_hashes_verified': True,
              'disjoint_splits_verified': True, 'test_performance_absent': True,
              'plot_csv_matches_independent_predictions': True,
              'selected_models_match_validation_selection': True,
              'independent_implementation': 'Polynomial evaluation, direct recurrence, and numpy convolution; experiment module not imported.',
              'fir_vs_first_order': {'relative_rmse_reduction_percent': float(100*(1-f[fir].mean()/f[first].mean())),
                                     'absolute_mean_rmse_reduction': float(diff.mean()),
                                     'descriptive_paired_bootstrap_ci': ci.tolist(),
                                     'flights_improved': int((diff > 0).sum()),
                                     'note': 'Validation was also used for selection; interval is descriptive, not final-test inference.'},
              'verification_script_sha256': digest(__file__),
              'environment_packages': {k: importlib.metadata.version(k) for k in ['numpy', 'scipy', 'pandas', 'matplotlib']}}
    (out/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
