"""Pelican E1: raw-data audit and causal commanded-to-reported motor models.

This experiment does not train SINDy or evaluate MPC. No final test predictions
are produced. Training losses and validation summaries weight flights equally.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, default=convert, allow_nan=False), encoding='utf-8')


def make_splits(n, config):
    order = np.random.default_rng(config['split_seed']).permutation(n) + 1
    a, b = config['train_flights'], config['validation_flights']
    return {'train': sorted(order[:a].tolist()),
            'validation': sorted(order[a:a+b].tolist()),
            'test': sorted(order[a+b:].tolist()),
            'seed': config['split_seed'],
            'grouping': 'Whole MAT cells; no dates/battery/session identifiers present.',
            'test_usage': 'Schema and integrity audit only; no fits, selection, diagnostics or performance.'}


def load_and_audit(path, config, out):
    data = loadmat(path, simplify_cells=True)
    raw = data['flights']
    if not isinstance(raw, list) or len(raw) != 54:
        raise ValueError('Unexpected Pelican structure; revisit split before analysis.')
    splits = make_splits(len(raw), config)
    write_json(out / 'splits.json', splits)
    lookup = {i: split for split in ['train', 'validation', 'test'] for i in splits[split]}
    schema, quality, episodes = [], [], {}
    fingerprints = {}
    for i, flight in enumerate(raw, 1):
        n = int(flight['len'])
        for field, value in flight.items():
            a = np.asarray(value)
            schema.append({'flight_id': i, 'split': lookup[i], 'field': field,
                           'shape': str(a.shape), 'dtype': str(a.dtype),
                           'nonfinite_count': int((~np.isfinite(a)).sum())})
        u, y = np.asarray(flight['Motors_CMD']), np.asarray(flight['Motors'])
        if u.shape != (n, 4) or y.shape != (n, 4):
            raise ValueError(f'Flight {i}: unexpected motor dimensions')
        if not (np.isfinite(u).all() and np.isfinite(y).all()):
            raise ValueError(f'Flight {i}: non-finite motor signal; no silent imputation')
        vel = flight['Vel']
        vel_error = float(np.max(np.abs(np.diff(flight['Pos'], axis=0) / config['dt'] - vel)))
        er = flight['Euler_Rates']
        euler_error = float(np.max(np.abs(np.diff(flight['Euler'], axis=0) / config['dt'] - er)))
        fingerprint = hashlib.sha256(u.tobytes() + y.tobytes()).hexdigest()
        duplicate = fingerprints.get(fingerprint)
        fingerprints[fingerprint] = i
        quality.append({'flight_id': i, 'split': lookup[i], 'motor_samples': n,
                        'duration_s': (n-1)*config['dt'], 'motor_signal_sha256': fingerprint,
                        'duplicate_of': duplicate, 'velocity_rows': len(vel),
                        'pqr_rows': len(flight['pqr']),
                        'pqr_last_row_all_zero': bool(np.all(flight['pqr'][-1] == 0)),
                        'velocity_forward_difference_max_abs_error': vel_error,
                        'euler_rate_forward_difference_max_abs_error': euler_error})
        # Reserved test motor arrays are not passed into any model/diagnostic function.
        if lookup[i] != 'test':
            episodes[i] = {'id': i, 'split': lookup[i], 'u': u, 'y': y}
    if any(q['duplicate_of'] is not None for q in quality):
        raise ValueError('Exact duplicated flights detected; group before splitting.')
    pd.DataFrame(schema).to_csv(out / 'field_audit.csv', index=False)
    pd.DataFrame(quality).to_csv(out / 'flight_audit.csv', index=False)
    audit = {'flight_count': len(raw), 'total_motor_rows': sum(q['motor_samples'] for q in quality),
             'dt_s': config['dt'], 'time_axis': 'Reconstructed from documented 100 Hz; no timestamps in MAT.',
             'raw_fields': sorted(raw[0]), 'excluded_flights': [],
             'pqr_zero_padded_flights': [q['flight_id'] for q in quality if q['pqr_last_row_all_zero']],
             'velocity_max_difference_error': max(q['velocity_forward_difference_max_abs_error'] for q in quality),
             'euler_rates_max_difference_error': max(q['euler_rate_forward_difference_max_abs_error'] for q in quality),
             'upstream_statistics_used': False,
             'warnings': ['Motors are vendor-reported estimates; not independently measured shaft speed.',
                          'Motor coding is not rad/s or thrust. No cross-signal unit equality assumed.',
                          'Upstream smoothing and sensor synchronization limit physical-delay inference.',
                          'Vel/Euler_Rates are forward differences: row k uses state k+1.',
                          'The pqr terminal padding must not become a physical angular-rate sample.',
                          'Only flight-level grouping is possible; independence across sessions is unverified.']}
    write_json(out / 'data_audit.json', audit)
    return episodes, splits, audit


def train_scales(episodes, score_start):
    mu = np.mean([e['u'][score_start:].mean(axis=0) for e in episodes], axis=0)
    var = np.mean([np.mean((e['u'][score_start:]-mu)**2, axis=0) for e in episodes], axis=0)
    return mu, np.sqrt(np.maximum(var, 1e-12))


def history_design(u, h, start, motor, mu, scale):
    if start < h-1:
        raise ValueError('Insufficient causal command history')
    idx = np.arange(start, len(u))
    z = np.column_stack([(u[idx-j, motor]-mu[motor])/scale[motor] for j in range(h)])
    return np.column_stack([np.ones(len(idx)), z])


def fit_command_models(train, config, mu, scale):
    s = config['prediction_start'] + config['warmup_samples']
    hmax = max(config['fir_history_samples'])
    grams = np.zeros((4, hmax+1, hmax+1))
    rhs = np.zeros((4, hmax+1))
    qgrams, qrhs = np.zeros((4, 3, 3)), np.zeros((4, 3))
    for e in train:
        for m in range(4):
            x = history_design(e['u'], hmax, s, m, mu, scale)
            y = e['y'][s:, m]
            grams[m] += (x.T@x)/len(y)/len(train)
            rhs[m] += (x.T@y)/len(y)/len(train)
            qx = np.column_stack([x[:, :2], x[:, 1]**2])
            qgrams[m] += (qx.T@qx)/len(y)/len(train)
            qrhs[m] += (qx.T@y)/len(y)/len(train)
    models = []
    for degree in [1, 2]:
        coefficients = []
        for m in range(4):
            g, r = (grams[m, :2, :2], rhs[m, :2]) if degree == 1 else (qgrams[m], qrhs[m])
            coefficients.append(np.linalg.lstsq(g, r, rcond=None)[0])
        models.append({'name': 'static_linear' if degree == 1 else 'static_quadratic',
                       'kind': 'static', 'degree': degree, 'coefficients': np.array(coefficients),
                       'mu': mu, 'scale': scale})
    for h in config['fir_history_samples']:
        for alpha in config['ridge_alphas']:
            penalty = np.diag(np.r_[0.0, np.ones(h)])
            coef = [np.linalg.solve(grams[m, :h+1, :h+1]+alpha*penalty, rhs[m, :h+1]) for m in range(4)]
            models.append({'name': f'fir_h{h}_a{alpha:g}', 'kind': 'fir', 'history': h,
                           'alpha': alpha, 'coefficients': np.array(coef), 'mu': mu, 'scale': scale})
    return models


def response_features(u, initial_y, pole, delay, start):
    """Features for y[k]=a*y[k-1]+(1-a)*(gain*u[k-delay]+bias)."""
    if delay < 0 or start < delay or start < 1 or not 0 <= pole < 1:
        raise ValueError('Invalid causal response settings')
    z = u[start-delay:len(u)-delay] if delay else u[start:]
    filtered = lfilter([1-pole], [1, -pole], z, axis=0)
    transient = pole ** np.arange(1, len(z)+1, dtype=float)
    bias = 1-transient
    initial = transient[:, None]*np.asarray(initial_y)[None, :]
    return filtered, bias, initial


def fit_first_order(train, pole, delay, config):
    start, warm = config['prediction_start'], config['warmup_samples']
    gram, rhs, yy = np.zeros((4, 2, 2)), np.zeros((4, 2)), np.zeros(4)
    for e in train:
        z, b, initial = response_features(e['u'], e['y'][start-1], pole, delay, start)
        z, b = z[warm:], b[warm:]
        target = e['y'][start+warm:]-initial[warm:]
        w = 1/len(target)/len(train)
        gram[:, 0, 0] += np.sum(z*z, axis=0)*w
        gram[:, 0, 1] += (z.T@b)*w
        gram[:, 1, 0] += (z.T@b)*w
        gram[:, 1, 1] += (b@b)*w
        rhs[:, 0] += np.sum(z*target, axis=0)*w
        rhs[:, 1] += (target.T@b)*w
        yy += np.sum(target*target, axis=0)*w
    coef = np.array([np.linalg.lstsq(gram[m], rhs[m], rcond=None)[0] for m in range(4)])
    # Positive static gain; if the unconstrained optimum is negative, use boundary gain=0.
    for m in range(4):
        if coef[m, 0] < 0:
            coef[m] = [0.0, rhs[m, 1]/gram[m, 1, 1]]
    mse = yy - 2*np.sum(coef*rhs, axis=1) + np.einsum('mi,mij,mj->m', coef, gram, coef)
    return coef, np.maximum(mse, 0)


def predict(model, episode, config):
    start = config['prediction_start']
    u = episode['u']
    yhat = np.empty((len(u)-start, 4))
    if model['kind'] == 'first_order':
        for m in range(4):
            z, b, initial = response_features(u[:, m:m+1], episode['y'][start-1, m:m+1],
                                              model['poles'][m], model['delay'], start)
            gain, bias = model['coefficients'][m]
            yhat[:, m] = gain*z[:, 0] + bias*b + initial[:, 0]
    else:
        for m in range(4):
            if model['kind'] == 'static':
                z = (u[start:, m]-model['mu'][m])/model['scale'][m]
                x = np.column_stack([z**j for j in range(model['degree']+1)])
            else:
                x = history_design(u, model['history'], start, m, model['mu'], model['scale'])
            yhat[:, m] = x@model['coefficients'][m]
    if not np.isfinite(yhat).all():
        raise FloatingPointError('Nonfinite prediction')
    return yhat


def evaluate(model, episodes, config):
    start, warm = config['prediction_start'], config['warmup_samples']
    rows = []
    for e in episodes:
        prediction = predict(model, e, config)[warm:]
        truth = e['y'][start+warm:]
        for m in range(4):
            residual = prediction[:, m]-truth[:, m]
            rmse = float(np.sqrt(np.mean(residual**2)))
            rows.append({'model': model['name'], 'flight_id': e['id'], 'split': e['split'],
                         'motor': m+1, 'samples': len(truth), 'rmse_code_units': rmse,
                         'mae_code_units': float(np.mean(np.abs(residual))),
                         'bias_code_units': float(np.mean(residual)),
                         'nrmse_by_flight_sd': rmse/max(float(np.std(truth[:, m])), 1e-12)})
    return rows


def diagnostics(episodes, config, out):
    rows, correlations = [], []
    s, maxlag = config['prediction_start']+config['warmup_samples'], config['correlation_max_lag']
    for e in episodes:
        if e['split'] != 'train':
            continue
        u, y = e['u'][s:], e['y'][s:]
        du, dy = np.diff(u, axis=0), np.diff(y, axis=0)
        for m in range(4):
            row = {'flight_id': e['id'], 'split': 'train', 'motor': m+1,
                   'u_min': float(u[:, m].min()), 'u_max': float(u[:, m].max()),
                   'u_std': float(u[:, m].std()), 'y_min': float(y[:, m].min()),
                   'y_max': float(y[:, m].max()), 'y_std': float(y[:, m].std()),
                   'du_rms': float(np.sqrt(np.mean(du[:, m]**2))),
                   'fraction_near_0': float(np.mean(u[:, m] < 0.5)),
                   'fraction_near_218': float(np.mean(u[:, m] > 217.5))}
            scores = []
            for lag in range(-maxlag, maxlag+1):
                # Positive lag: reported response follows command; negative is diagnostic only.
                x = du[:len(du)-lag, m] if lag > 0 else du[-lag:, m]
                z = dy[lag:, m] if lag > 0 else dy[:len(dy)+lag, m]
                x, z = x-x.mean(), z-z.mean()
                den = np.linalg.norm(x)*np.linalg.norm(z)
                corr = float(x@z/den) if den > 0 else 0.0
                scores.append(corr)
                correlations.append({'flight_id': e['id'], 'motor': m+1,
                                     'lag_s': lag*config['dt'], 'difference_correlation': corr})
            row.update(peak_lag_s=(int(np.argmax(scores))-maxlag)*config['dt'],
                       peak_difference_correlation=float(max(scores)))
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'motor_diagnostics.csv', index=False)
    pd.DataFrame(correlations).to_csv(out / 'lag_correlations.csv', index=False)
    return pd.DataFrame(rows), pd.DataFrame(correlations)


def summarize(metrics, config):
    perflight = metrics.groupby(['model', 'flight_id'])['rmse_code_units'].mean().unstack(0)
    base = perflight['static_linear'].to_numpy()
    rng = np.random.default_rng(config['bootstrap_seed'])
    indices = rng.integers(0, len(base), size=(config['bootstrap_replicates'], len(base)))
    rows = []
    for name in perflight.columns:
        values = perflight[name].to_numpy()
        difference = base-values
        ci = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
        rows.append({'model': name, 'validation_flights': len(values),
                     'mean_flight_motor_rmse': float(values.mean()),
                     'median_flight_motor_rmse': float(np.median(values)),
                     'change_vs_static_percent': float(100*(values.mean()/base.mean()-1)),
                     'paired_rmse_reduction_ci_low': float(ci[0]),
                     'paired_rmse_reduction_ci_high': float(ci[1]),
                     'flights_better_than_static': int(np.sum(values < base))})
    return pd.DataFrame(rows), perflight


def figures(models, validation, metrics, correlations, config, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#777777', '#E69F00', '#0072B2', '#009E73']
    labels = ['Static linear', 'Static quadratic', 'First-order response', 'Command-history FIR']
    names = [m['name'] for m in models]
    folder = out / 'figures'
    folder.mkdir()
    def save(fig, name):
        fig.savefig(folder / f'{name}.pdf', bbox_inches='tight')
        fig.savefig(folder / f'{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    agg = metrics.groupby(['flight_id', 'model'])['rmse_code_units'].mean().unstack()
    fig, ax = plt.subplots(figsize=(7.0, 3.1), layout='constrained')
    for _, row in agg.iterrows():
        ax.plot(range(4), row[names], color='0.78', lw=.8, marker='o', ms=3)
    ax.plot(range(4), agg[names].mean(), color='#0072B2', marker='D', lw=2, ms=6, label='Mean across flights')
    ax.set_xticks(range(4), ['Static\nlinear', 'Static\nquadratic', 'First-order\nresponse', 'Command-history\nFIR'])
    ax.set_ylabel('Mean motor RMSE (reported-speed code units)')
    ax.set_title('Validation flights; full-flight predictions after common warmup')
    ax.legend(frameon=False)
    save(fig, 'validation_model_comparison')
    rank = agg['static_linear'].sort_values()
    typical = int(rank.index[len(rank)//2])
    e = next(e for e in validation if e['id'] == typical)
    start, warm = config['prediction_start'], config['warmup_samples']
    scorestart = start+warm
    window = int(5/config['dt'])
    # Choose the 5-second validation window by command activity, not prediction quality.
    energy = np.sum(np.diff(e['u'], axis=0)**2, axis=1)
    candidates = np.arange(scorestart, len(e['u'])-window, window)
    begin = int(max(candidates, key=lambda k: energy[k:k+window-1].sum()))
    stop = begin+window
    predictions = {m['name']: predict(m, e, config) for m in models}
    time_axis = np.arange(begin, stop)*config['dt']
    plot_data = {'t_s': time_axis}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True, layout='constrained')
    for m, ax in enumerate(axes.flat):
        ax.plot(time_axis, e['y'][begin:stop, m], color='black', lw=1.3, label='Reported speed')
        plot_data[f'm{m+1}_reported'] = e['y'][begin:stop, m]
        plot_data[f'm{m+1}_command'] = e['u'][begin:stop, m]
        for model, label, color in zip(models, labels, colors):
            y = predictions[model['name']][begin-start:stop-start, m]
            ax.plot(time_axis, y, color=color, lw=1, linestyle='--' if model['kind'] == 'static' else '-', label=label)
            plot_data[f'm{m+1}_{model["name"]}'] = y
        ax.set_title(f'Motor {m+1}')
        ax.set_ylabel('Reported-speed code units')
        ax.set_xlabel('Time (s)')
    handles, leglabels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, leglabels, loc='outside upper center', ncol=3, frameon=False)
    save(fig, 'representative_response')
    pd.DataFrame(plot_data).to_csv(out / 'representative_response.csv', index=False)
    # Raw signal units are displayed on separate panels, without imposing equal scales.
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.9), sharex=True, layout='constrained')
    for m, color in enumerate(['#0072B2', '#E69F00', '#009E73', '#CC79A7']):
        axes[0].plot(time_axis, e['u'][begin:stop, m], color=color, lw=1, label=f'Motor {m+1}')
        axes[1].plot(time_axis, e['y'][begin:stop, m], color=color, lw=1)
    axes[0].set_ylabel('Command code units')
    axes[1].set_ylabel('Reported-speed code units')
    axes[1].set_xlabel('Time (s)')
    axes[0].legend(ncol=4, frameon=False)
    save(fig, 'command_and_reported_speed')
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.3), sharex=True, sharey=True, layout='constrained')
    for m, ax in enumerate(axes.flat, 1):
        g = correlations[correlations.motor == m].groupby('lag_s').difference_correlation
        mid, low, high = g.median(), g.quantile(.25), g.quantile(.75)
        ax.plot(mid.index*1000, mid, color='#0072B2')
        ax.fill_between(mid.index*1000, low, high, color='#0072B2', alpha=.18)
        ax.axvline(0, color='0.5', ls='--', lw=.7)
        ax.set_title(f'Motor {m}')
        ax.set_xlabel('Response lag relative to command (ms)')
        ax.set_ylabel('Correlation of first differences')
    save(fig, 'training_lag_diagnostics')
    return {'representative_flight': typical, 'window_start_s': begin*config['dt'],
            'window_end_s': stop*config['dt'],
            'selection': 'Median validation flight by static-linear RMSE; highest-command-activity nonoverlapping 5-s window.'}


def run(args):
    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    out = args.run_dir
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', config)
    raw_provenance = json.loads(args.raw.with_name('provenance.json').read_text(encoding='utf-8'))
    assert sha256(args.raw) == raw_provenance['sha256'], 'Raw-file hash mismatch'
    write_json(out / 'raw_provenance.json', raw_provenance)
    source_files = [Path(__file__), ROOT/'scripts/fetch_pelican.py', args.config]
    write_json(out / 'code_manifest.json', {str(p.relative_to(ROOT)): sha256(p) for p in source_files})
    write_json(out / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
                                         'numpy': np.__version__, 'pandas': pd.__version__,
                                         'started_utc': datetime.now(timezone.utc).isoformat()})
    log_path = out / 'progress.jsonl'
    def log(event, **kwargs):
        row = {'elapsed_s': round(time.monotonic()-started, 2), 'event': event, **kwargs}
        line = json.dumps(row)
        with log_path.open('a', encoding='utf-8') as f:
            f.write(line+'\n')
        print(line, flush=True)
        if time.monotonic()-started > config['runtime_limit_s']:
            raise TimeoutError('Experiment exceeded configured wall-clock limit')
    log('raw_file_verified')
    episodes, splits, audit = load_and_audit(args.raw, config, out)
    train = [episodes[i] for i in splits['train']]
    validation = [episodes[i] for i in splits['validation']]
    log('data_audited', flights=audit['flight_count'], samples=audit['total_motor_rows'])
    diagnostic_rows, correlations = diagnostics(train, config, out)
    log('training_diagnostics_complete')
    mu, scale = train_scales(train, config['prediction_start']+config['warmup_samples'])
    commands = fit_command_models(train, config, mu, scale)
    log('command_models_fitted', candidates=len(commands))
    first_orders, pole_search = [], []
    for delay in config['delay_samples']:
        best_loss = np.full(4, np.inf)
        best_coef, best_pole, best_tau = np.zeros((4, 2)), np.zeros(4), np.zeros(4)
        for tau in config['tau_grid_s']:
            pole = float(np.exp(-config['dt']/tau)) if tau > 0 else 0.0
            coef, loss = fit_first_order(train, pole, delay, config)
            for m in range(4):
                pole_search.append({'delay_samples': delay, 'tau_grid_s': tau, 'motor': m+1,
                                    'training_mean_flight_mse': loss[m]})
            improved = loss < best_loss
            best_loss[improved], best_coef[improved] = loss[improved], coef[improved]
            best_pole[improved], best_tau[improved] = pole, tau
        first_orders.append({'name': f'first_order_delay{delay}', 'kind': 'first_order', 'delay': delay,
                             'poles': best_pole, 'tau_s': best_tau, 'coefficients': best_coef,
                             'training_mean_flight_mse': best_loss})
        log('first_order_delay_fitted', delay_samples=delay, tau_s=best_tau.tolist())
    pd.DataFrame(pole_search).to_csv(out / 'training_pole_search.csv', index=False)
    # Freeze every candidate before any validation-based choice.
    all_models = commands+first_orders
    write_json(out / 'all_candidate_models.json', all_models)
    candidates, candidate_rows = [], []
    for model in all_models:
        rows = evaluate(model, validation, config)
        score = float(np.mean([r['rmse_code_units'] for r in rows]))
        candidates.append({'model': model['name'], 'kind': model['kind'], 'validation_score': score})
        candidate_rows.extend(rows)
    pd.DataFrame(candidates).to_csv(out / 'validation_model_selection.csv', index=False)
    pd.DataFrame(candidate_rows).to_csv(out / 'validation_all_candidates.csv', index=False)
    scores = {r['model']: r['validation_score'] for r in candidates}
    selected_first = min(first_orders, key=lambda model: scores[model['name']])
    selected_fir = min([m for m in commands if m['kind'] == 'fir'], key=lambda model: scores[model['name']])
    selected = commands[:2]+[selected_first, selected_fir]
    write_json(out / 'motor_response_models.json', selected)
    log('models_selected', first_order=selected_first['name'], fir=selected_fir['name'])
    metric_rows = [r for r in candidate_rows if r['model'] in [m['name'] for m in selected]]
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / 'validation_per_flight_motor.csv', index=False)
    summary, perflight = summarize(metrics, config)
    summary.to_csv(out / 'validation_summary.csv', index=False)
    figure_info = figures(selected, validation, metrics, correlations, config, out)
    peak_summary = diagnostic_rows.groupby('motor')[['peak_lag_s', 'peak_difference_correlation']].median().to_dict('index')
    report = {'status': 'completed', 'experiment': 'Pelican E1', 'raw_sha256': raw_provenance['sha256'],
              'data': audit, 'splits': splits, 'selection_set': 'validation',
              'final_test_performance_computed': False, 'score_units': 'Reported-speed vendor code units',
              'score_definition': 'Mean over flights of mean over 4 scalar motor RMSEs; not a 4-vector RMSE.',
              'first_order': selected_first, 'fir': selected_fir,
              'validation_results': summary.to_dict('records'), 'training_correlation_peaks_median': peak_summary,
              'figures': figure_info, 'elapsed_s': time.monotonic()-started,
              'limitations': ['Validation data also selected delay/history/regularization; results are developmental, not final test claims.',
                              'Flight bootstrap intervals are descriptive after selection and assume exchangeable flights; session IDs absent.',
                              'Model tau/delay describe preprocessed reported signals, not identified physical motor constants.',
                              'Positive/negative cross-correlation peaks are descriptive; no negative-delay predictive model is allowed.',
                              'No SINDy identification, trajectory control, or new flight experiment was performed.']}
    write_json(out / 'report.json', report)
    # Report text intentionally produced from computed outputs, with explicit experiment boundary.
    lines = ['# Pelican E1 motor-response experiment', '', '## Material Passport',
             '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run',
             '- Verification Status: ANALYZED (see verification.json for independent artifact checks)',
             '- Version Label: pelican_e1_v1', '',
             f'Raw flights: {audit["flight_count"]}; motor samples: {audit["total_motor_rows"]}; documented dt: {config["dt"]} s.',
             f'Train / validation / reserved test: {len(train)} / {len(validation)} / {len(splits["test"])}.',
             'Only training/validation model predictions were computed. Test flights received schema/integrity checks only.', '',
             '## Validation results (also used for model selection)', '',
             '| Model | Mean scalar-motor RMSE | Change vs static linear | Flights better than static |',
             '|---|---:|---:|---:|']
    for r in summary.to_dict('records'):
        lines.append(f'| {r["model"]} | {r["mean_flight_motor_rmse"]:.6f} | {r["change_vs_static_percent"]:.2f}% | {r["flights_better_than_static"]} / {len(validation)} |')
    lines += ['', 'Units are vendor-reported speed codes, not RPM or rad/s. Lower error is better.', '',
              '## Effective response parameters', '',
              f'First-order selected command delay: {selected_first["delay"]*config["dt"]:.3f} s.',
              f'Per-motor training-selected grid time constants: {selected_first["tau_s"].tolist()} s.',
              f'FIR history: {selected_fir["history"]} samples, spanning {(selected_fir["history"]-1)*config["dt"]:.3f} s.',
              'These are effective parameters of preprocessed telemetry, not calibrated physical motor constants.', '',
              '## Alignment and limitations', '', *['- '+x for x in audit['warnings']],
              *['- '+x for x in report['limitations']], '',
              '## Figures', '',
              '![Command and speed](figures/command_and_reported_speed.png)',
              '![Validation comparison](figures/validation_model_comparison.png)',
              '![Representative response](figures/representative_response.png)',
              '![Training lag diagnostics](figures/training_lag_diagnostics.png)', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    log('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT/'data/pelican/raw/AscTec_Pelican_Flight_Dataset.mat')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/pelican_e1.json')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/pelican_e1_v1')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)
