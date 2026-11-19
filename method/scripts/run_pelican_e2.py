"""Run Pelican E2 in separate fit/freeze and final-test stages."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from pelican_e2.data import prepare_flight, target_acceleration, advance
from pelican_e2.model import (scales_from_training, training_statistics, fit_candidate, evaluate)


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False), encoding='utf-8')


def restore_model(model):
    for key in ['mu', 'scale', 'y_mu', 'y_scale', 'feature_scale', 'coefficients']:
        if key in model:
            model[key] = np.asarray(model[key], dtype=float)
    return model


def prepare(raw, ids, split, config, motor_model, out):
    result, audit = [], []
    for fid in ids:
        e = prepare_flight(raw[fid-1], fid, config['dt'], motor_model)
        e['split'] = split
        result.append(e)
        np.savez_compressed(out/'data'/f'flight_{fid:02d}.npz',
                            **{k: e[k] for k in ['x', 'u', 'motor_hat', 't', 'raw_index']})
        # Exact kinematic reconstruction when known next increments are supplied,
        # solely an alignment/unit check; these oracle targets are never a predictor.
        idx = np.arange(0, len(e['x'])-1, 101)
        y = target_acceleration(e, idx, config)
        exact = advance(e['x'][idx], y, config)
        truth = e['x'][idx+1]
        sign = np.where(np.sum(exact[:, 6:10]*truth[:, 6:10], axis=1) < 0, -1, 1)
        audit.append({'flight_id': fid, 'split': split, 'states': len(e['x']),
                      'position_reconstruction_max_abs': float(np.max(np.abs(exact[:, :3]-truth[:, :3]))),
                      'velocity_reconstruction_max_abs': float(np.max(np.abs(exact[:, 3:6]-truth[:, 3:6]))),
                      'quaternion_reconstruction_max_abs': float(np.max(np.abs(exact[:, 6:10]*sign[:, None]-truth[:, 6:10]))),
                      'omega_reconstruction_max_abs': float(np.max(np.abs(exact[:, 10:13]-truth[:, 10:13])))})
    return result, audit


def main(args):
    out = args.run_dir
    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.stage == 'fit':
        out.mkdir(parents=True, exist_ok=False)
        for folder in ['data', 'models', 'validation', 'test']:
            (out/folder).mkdir()
        write_json(out/'config.json', config)
    else:
        assert config == json.loads((out/'config.json').read_text()), 'Config differs from frozen run'
    def log(event, **kwargs):
        record = {'utc': datetime.now(timezone.utc).isoformat(), 'stage': args.stage,
                  'elapsed_s': round(time.monotonic()-started, 3), 'event': event, **kwargs}
        line = json.dumps(record)
        with (out/'progress.jsonl').open('a', encoding='utf-8') as f: f.write(line+'\n')
        print(line, flush=True)
        if time.monotonic()-started > config['runtime_limit_s']:
            raise TimeoutError('Configured E2 stage runtime exceeded')
    e1 = ROOT/'runs/pelican_e1_v1'
    raw_path = ROOT/'data/pelican/raw/AscTec_Pelican_Flight_Dataset.mat'
    raw_expected = json.loads((e1/'raw_provenance.json').read_text())['sha256']
    assert checksum(raw_path) == raw_expected
    splits = json.loads((e1/'splits.json').read_text())
    motor_file = e1/'motor_response_models.json'
    motor_model = next(m for m in json.loads(motor_file.read_text()) if m['kind'] == 'first_order')
    assert motor_model['delay'] == 0, 'This adapter requires the E1 selected zero-extra-delay model'
    if args.stage == 'fit':
        write_json(out/'splits.json', {**splits, 'test_usage': 'Reserved until frozen final evaluation stage.'})
        write_json(out/'motor_model.json', motor_model)
        write_json(out/'environment.json', {'python': sys.version, 'platform': platform.platform(),
                                           'packages': {k: importlib.metadata.version(k) for k in ['numpy','scipy','pandas','pysindy','matplotlib']}})
        log('training_validation_load', final_test_processed=False)
        raw = loadmat(raw_path, simplify_cells=True)['flights']
        train, audit_train = prepare(raw, splits['train'], 'train', config, motor_model, out)
        validation, audit_val = prepare(raw, splits['validation'], 'validation', config, motor_model, out)
        del raw
        pd.DataFrame(audit_train+audit_val).to_csv(out/'data/alignment_audit.csv', index=False)
        log('training_validation_prepared', train_flights=len(train), validation_flights=len(validation))
        scaling = scales_from_training(train, config)
        g, c, samples = training_statistics(train, config, scaling)
        motor_scaling = scales_from_training(train, config, motor=True)
        gm, cm, _ = training_statistics(train, config, motor_scaling, motor=True)
        np.savez_compressed(out/'models/training_statistics.npz', command_gram=g, command_rhs=c,
                            motor_gram=gm, motor_rhs=cm)
        log('training_statistics_ready', training_transitions=samples, max_features=len(g))
        specifications = []
        for family, lags_choices, motor in [('sindy_current', [[]], False),
                                            ('sindy_history', config['history_lags'], False),
                                            ('motor_sindy', [[]], True)]:
            for lags in lags_choices:
                for alpha in config['sindy_alphas']:
                    for threshold in config['sindy_thresholds']:
                        specifications.append((family, lags, 2, alpha, threshold, motor))
        for lags in config['history_lags']:
            for alpha in config['linear_alphas']:
                specifications.append(('linear_history', lags, 1, alpha, 0.0, False))
        candidates, selection_rows, validation_rows = [], [], []
        for number, (family, lags, degree, alpha, threshold, motor) in enumerate(specifications, 1):
            model = fit_candidate(gm if motor else g, cm if motor else c,
                                  motor_scaling if motor else scaling,
                                  lags, degree, alpha, threshold, config, motor)
            model['family'] = family
            model['name'] = f'{family}_h{max(lags,default=0)}_a{alpha:g}_t{threshold:g}'
            write_json(out/'models'/f'{model["name"]}.json', model)
            rows, _, _, score = evaluate(model, validation, motor_model, config)
            selection_rows.append({'candidate': model['name'], 'family': family, 'lags': json.dumps(lags),
                                   'degree': degree, 'alpha': alpha, 'threshold': threshold,
                                   'primary_failure_rate': score[0], 'primary_velocity_rmse': score[1],
                                   'nonzero_coefficients': model['nonzero_coefficients'],
                                   'feature_count': model['feature_count']})
            validation_rows.extend(rows)
            candidates.append((model, score))
            pd.DataFrame(selection_rows).to_csv(out/'validation/model_selection.csv', index=False)
            log('candidate_evaluated', candidate=number, total=len(specifications), model=model['name'],
                failure_rate=score[0], velocity_rmse=score[1], nonzero=model['nonzero_coefficients'])
        selected = []
        for family in ['sindy_current','linear_history','sindy_history','motor_sindy']:
            model, score = min([(m,s) for m,s in candidates if m['family'] == family],
                               key=lambda pair: (pair[1][0], pair[1][1], pair[0]['nonzero_coefficients']))
            selected.append({**model, 'candidate_name': model['name'], 'name': family,
                             'validation_score': list(score)})
        history_lags = next(m['lags'] for m in selected if m['family'] == 'sindy_history')
        matched, score = min([(m,s) for m,s in candidates if m['family'] == 'linear_history' and m['lags'] == history_lags],
                             key=lambda pair: pair[1])
        selected.append({**matched, 'candidate_name': matched['name'], 'name': 'linear_history_matched',
                         'family': 'linear_history_matched', 'validation_score': list(score)})
        selected.append({'name': 'constant_twist', 'family': 'constant_twist', 'lags': [], 'motor': False,
                         'nonzero_coefficients': 0, 'feature_count': 0})
        pd.DataFrame(validation_rows).to_csv(out/'validation/all_candidates_per_flight.csv', index=False)
        write_json(out/'models/selected_models.json', selected)
        code_paths = list((ROOT/'pelican_e2').glob('*.py'))+[Path(__file__), args.config]
        source = {str(p.relative_to(ROOT)).replace('\\','/'): checksum(p) for p in code_paths}
        freeze = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'raw_sha256': raw_expected,
                  'e1_motor_models_sha256': checksum(motor_file), 'code_sha256': source,
                  'selected_models_sha256': checksum(out/'models/selected_models.json'),
                  'config_sha256': checksum(out/'config.json'), 'splits_sha256': checksum(out/'splits.json'),
                  'motor_model_sha256': checksum(out/'motor_model.json'),
                  'selection_rule': 'Lexicographic primary-horizon failure rate, macro velocity RMSE, nonzero count.',
                  'final_test_processed_before_freeze': False,
                  'training_transitions': samples,
                  'predeclared_plot_selection': 'Median and worst test flight by current-SINDy 0.5s velocity RMSE; median-error origin within each.'}
        write_json(out/'FROZEN.json', freeze)
        log('models_frozen', selected=[{'model': m['name'], 'candidate': m.get('candidate_name')} for m in selected])
    else:
        if (out/'test/COMPLETED.json').exists():
            raise RuntimeError('Final test already completed; use verifier, not another test run.')
        freeze = json.loads((out/'FROZEN.json').read_text())
        for path, expected in freeze['code_sha256'].items():
            assert checksum(ROOT/path) == expected, 'Source changed after freezing: '+path
        for key, path in [('selected_models_sha256','models/selected_models.json'),('config_sha256','config.json'),
                          ('splits_sha256','splits.json'),('motor_model_sha256','motor_model.json')]:
            assert checksum(out/path) == freeze[key]
        assert checksum(motor_file) == freeze['e1_motor_models_sha256']
        models = [restore_model(m) for m in json.loads((out/'models/selected_models.json').read_text())]
        log('frozen_models_verified_before_test_load')
        raw = loadmat(raw_path, simplify_cells=True)['flights']
        test, audit = prepare(raw, splits['test'], 'test', config, motor_model, out)
        del raw
        pd.DataFrame(audit).to_csv(out/'test/alignment_audit.csv', index=False)
        all_rows, all_windows = [], []
        for model in models:
            rows, windows, saved, score = evaluate(model, test, motor_model, config, save_predictions=True)
            all_rows.extend(rows); all_windows.extend(windows)
            endpoint_data = {}
            for entry in saved:
                prefix = f'f{entry["flight_id"]:02d}_s{round(entry["horizon_s"]/config["dt"]):03d}'
                for key in ['origins','prediction','truth','failed_at_step']:
                    endpoint_data[prefix+'_'+key] = entry[key]
            np.savez_compressed(out/'test'/f'{model["name"]}_endpoints.npz', **endpoint_data)
            pd.DataFrame(all_rows).to_csv(out/'test/prediction_per_flight.csv', index=False)
            log('test_model_complete', model=model['name'], failure_rate=score[0], velocity_rmse=score[1])
        pd.DataFrame(all_windows).to_csv(out/'test/prediction_per_window.csv', index=False)
        frame = pd.DataFrame(all_rows)
        summaries = []
        for (name, horizon), df in frame.groupby(['model', 'horizon_s']):
            row = {'model': name, 'horizon_s': horizon, 'test_flights': len(df),
                   'windows': int(df.windows.sum()), 'failed_windows': int(df.failed_windows.sum()),
                   'mean_flight_failure_rate': float(df.failure_rate.mean()),
                   'complete_all_flights': bool((df.failed_windows == 0).all())}
            for key in ['velocity','omega','position','attitude']:
                # Conditional on finite windows; completeness is explicitly alongside every mean.
                row[key+'_rmse_mean'] = float(df[key+'_rmse'].mean())
            summaries.append(row)
        pd.DataFrame(summaries).to_csv(out/'test/prediction_summary.csv', index=False)
        write_json(out/'test/COMPLETED.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
                                             'models_frozen_utc': freeze['frozen_utc'],
                                             'test_flights': splits['test'], 'models': [m['name'] for m in models],
                                             'source_changes_after_freeze': False,
                                             'model_results': summaries})
        log('final_test_complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['fit','test'], required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/pelican_e2.json')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/pelican_e2_v1')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args)
