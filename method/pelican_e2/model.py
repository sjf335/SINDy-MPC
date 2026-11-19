"""Sparse regression from exact weighted sufficient statistics; batch rollouts."""
import warnings
import numpy as np
from pysindy import STLSQ
from .data import state_features, target_acceleration, advance, qexp, qmultiply

PAIR_I, PAIR_J = np.triu_indices(13)
MEMORY_COLUMNS = np.r_[np.arange(6), np.arange(9, 13)]
CURRENT_SIZE = 1+13+len(PAIR_I)


def library(z, past, lags, degree):
    columns = [np.ones((len(z), 1)), z]
    if degree == 2:
        columns.append(z[:, PAIR_I]*z[:, PAIR_J])
    for lag in lags:
        previous = past[lag]
        columns.append(previous[:, MEMORY_COLUMNS]-z[:, MEMORY_COLUMNS])
        if degree == 2:
            columns.append(previous[:, 9:13]**2-z[:, 9:13]**2)
    return np.concatenate(columns, axis=1)


def feature_indices(lags, degree, maxlags):
    indices = list(range(CURRENT_SIZE if degree == 2 else 14))
    for lag in lags:
        start = CURRENT_SIZE+14*maxlags.index(lag)
        indices.extend(range(start, start+(14 if degree == 2 else 10)))
    return np.asarray(indices, dtype=int)


def scales_from_training(train, config, motor=False):
    means, seconds, ymeans, yseconds = [], [], [], []
    for e in train:
        idx = np.arange(config['training_start_index'], len(e['x'])-1, config['fit_stride'])
        z = np.c_[state_features(e['x'][idx], config['gravity']),
                  (e['motor_hat'] if motor else e['u'])[idx]]
        y = target_acceleration(e, idx, config)
        means.append(z.mean(0)); seconds.append((z*z).mean(0))
        ymeans.append(y.mean(0)); yseconds.append((y*y).mean(0))
    mu, ymu = np.mean(means, 0), np.mean(ymeans, 0)
    scale = np.sqrt(np.maximum(np.mean(seconds, 0)-mu*mu, 1e-6))
    yscale = np.sqrt(np.maximum(np.mean(yseconds, 0)-ymu*ymu, 1e-6))
    return {'mu': mu, 'scale': scale, 'y_mu': ymu, 'y_scale': yscale}


def training_statistics(train, config, scaling, motor=False):
    maxlags = [] if motor else config['history_lags'][-1]
    size = CURRENT_SIZE+14*len(maxlags)
    gram, rhs = np.zeros((size, size)), np.zeros((size, 6))
    samples = 0
    for e in train:
        zall = np.c_[state_features(e['x'], config['gravity']), e['motor_hat'] if motor else e['u']]
        zall = (zall-scaling['mu'])/scaling['scale']
        idx = np.arange(config['training_start_index'], len(e['x'])-1, config['fit_stride'])
        target = (target_acceleration(e, idx, config)-scaling['y_mu'])/scaling['y_scale']
        phi = library(zall[idx], {lag: zall[idx-lag] for lag in maxlags}, maxlags, 2)
        weight = 1/len(idx)/len(train)
        gram += (phi.T@phi)*weight
        rhs += (phi.T@target)*weight
        samples += len(idx)
    return gram, rhs, samples


def compressed_design(gram, rhs):
    """X.T@X=G and X.T@Y=C on the numerically supported feature subspace."""
    eig, vec = np.linalg.eigh((gram+gram.T)/2)
    mask = eig > max(float(eig.max())*1e-12, 1e-14)
    val, basis = eig[mask], vec[:, mask]
    x = np.sqrt(val)[:, None]*basis.T
    y = (basis.T@rhs)/np.sqrt(val)[:, None]
    return x, y, int(mask.sum())


def fit_candidate(gram, rhs, scaling, lags, degree, alpha, threshold, config, motor=False):
    maxlags = [] if motor else config['history_lags'][-1]
    idx = feature_indices(lags, degree, maxlags)
    g, c = gram[np.ix_(idx, idx)], rhs[idx]
    feature_scale = np.sqrt(np.maximum(np.diag(g), 1e-12))
    gn = g/feature_scale[:, None]/feature_scale[None, :]
    cn = c/feature_scale[:, None]
    x, y, rank = compressed_design(gn, cn)
    if degree == 1:
        coefficient = np.linalg.solve(gn+alpha*np.eye(len(gn)), cn).T
        warning_messages = []
    else:
        estimator = STLSQ(threshold=threshold, alpha=alpha, normalize_columns=False,
                          unbias=False, max_iter=40)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            estimator.fit(x, y)
        warning_messages = [str(w.message) for w in captured]
        coefficient = np.asarray(estimator.coef_)
    return {**scaling, 'lags': list(lags), 'degree': degree, 'alpha': alpha, 'threshold': threshold,
            'motor': motor, 'coefficients': coefficient, 'feature_scale': feature_scale,
            'rank': rank, 'nonzero_coefficients': int(np.count_nonzero(coefficient)),
            'feature_count': len(feature_scale), 'fit_warnings': warning_messages}


def context_for_episode(episode, origins, model, config):
    lags = model.get('lags', [])
    length = max(lags, default=0)+1
    indices = origins[:, None]-np.arange(length)[None, :]
    if indices.min() < 0:
        raise ValueError('History would cross a flight boundary')
    xhist = episode['x'][indices]
    input_hist = (episode['motor_hat'] if model.get('motor') else episode['u'])[indices]
    raw = np.concatenate([state_features(xhist, config['gravity']), input_hist], axis=-1)
    history = (raw-np.asarray(model['mu']))/np.asarray(model['scale']) if model.get('family') != 'constant_twist' else raw
    motor_prior = episode['motor_hat'][origins-1].copy()
    return episode['x'][origins].copy(), history.copy(), motor_prior


def rollout(initial, history, input_sequence, motor_prior, model, motor_model, config, keep_all=False):
    x = initial.copy()
    history = history.copy()
    motor_prior = motor_prior.copy()
    steps = input_sequence.shape[1]
    alive = np.ones(len(x), dtype=bool)
    failed_at = np.full(len(x), steps+1, dtype=int)
    wanted = {round(t/config['dt']) for t in config['horizons_s']}
    predictions = {}
    all_states = [x.copy()] if keep_all else None
    a = np.asarray(motor_model['poles'])
    gains, biases = np.asarray(motor_model['coefficients']).T
    for j in range(steps):
        u = input_sequence[:, j]
        if model.get('family') == 'constant_twist':
            next_x = x.copy()
            next_x[:, :3] += config['dt']*x[:, 3:6]
            next_x[:, 6:10] = qmultiply(x[:, 6:10], qexp(config['dt']*x[:, 10:13]))
            next_x[:, 6:10] /= np.linalg.norm(next_x[:, 6:10], axis=1, keepdims=True)
        else:
            if model['motor']:
                motor_prior = a*motor_prior+(1-a)*(gains*u+biases)
                inp = motor_prior
            else:
                inp = u
            z = (np.c_[state_features(x, config['gravity']), inp]-model['mu'])/model['scale']
            history[:, 0] = z
            phi = library(z, {lag: history[:, lag] for lag in model['lags']}, model['lags'], model['degree'])
            acceleration = (phi/model['feature_scale'])@model['coefficients'].T
            acceleration = acceleration*model['y_scale']+model['y_mu']
            next_x = advance(x, acceleration, config)
        valid = (np.isfinite(next_x).all(axis=1)
                 & (np.linalg.norm(next_x[:, 3:6], axis=1) <= config['max_speed_m_s'])
                 & (np.linalg.norm(next_x[:, 10:13], axis=1) <= config['max_body_rate_rad_s']))
        newly_failed = alive & ~valid
        failed_at[newly_failed] = j+1
        alive &= valid
        next_x[~alive] = initial[~alive]  # inert placeholder; every scored output is NaN after failure
        x = next_x
        if history.shape[1] > 1:
            history[:, 1:] = history[:, :-1].copy()
        if j+1 in wanted or keep_all:
            output = x.copy()
            output[~alive] = np.nan
            if j+1 in wanted:
                predictions[j+1] = output
            if keep_all:
                all_states.append(output)
    return predictions, failed_at, np.stack(all_states, axis=1) if keep_all else None


def origins_for(episode, config):
    maximum = round(max(config['horizons_s'])/config['dt'])
    return np.arange(config['origin_start_index'], len(episode['x'])-maximum, config['origin_stride'])


def evaluate(model, episodes, motor_model, config, save_predictions=False):
    rows, windows, saved = [], [], []
    nsteps = round(max(config['horizons_s'])/config['dt'])
    for e in episodes:
        origins = origins_for(e, config)
        x, history, motor_prior = context_for_episode(e, origins, model, config)
        inputs = e['u'][origins[:, None]+np.arange(nsteps)[None, :]]
        predictions, failure, _ = rollout(x, history, inputs, motor_prior, model, motor_model, config)
        for horizon in config['horizons_s']:
            steps = round(horizon/config['dt'])
            truth = e['x'][origins+steps]
            pred = predictions[steps]
            ok = failure > steps
            errors = {'position': np.linalg.norm(pred[:, :3]-truth[:, :3], axis=1),
                      'velocity': np.linalg.norm(pred[:, 3:6]-truth[:, 3:6], axis=1),
                      'omega': np.linalg.norm(pred[:, 10:13]-truth[:, 10:13], axis=1)}
            dot = np.sum(pred[:, 6:10]*truth[:, 6:10], axis=1)
            errors['attitude'] = 2*np.arccos(np.clip(np.abs(dot), 0, 1))
            row = {'model': model['name'], 'family': model['family'], 'flight_id': e['id'],
                   'horizon_s': horizon, 'windows': len(origins), 'failed_windows': int((~ok).sum()),
                   'failure_rate': float((~ok).mean())}
            for key, values in errors.items():
                row[key+'_rmse'] = float(np.sqrt(np.mean(values[ok]**2))) if ok.any() else None
            rows.append(row)
            if save_predictions:
                saved.append({'flight_id': e['id'], 'horizon_s': horizon, 'origins': origins,
                              'prediction': pred, 'truth': truth, 'failed_at_step': failure})
                for j, k in enumerate(origins):
                    windows.append({'model': model['name'], 'flight_id': e['id'], 'origin_index': int(k),
                                    'raw_origin_index': int(e['raw_index'][k]), 'horizon_s': horizon,
                                    'failed': bool(~ok[j]), **{key+'_error': float(val[j]) if ok[j] else None for key,val in errors.items()}})
    primary = [r for r in rows if abs(r['horizon_s']-config['primary_horizon_s']) < 1e-9]
    failure_rate = float(np.mean([r['failure_rate'] for r in primary]))
    values = [r['velocity_rmse'] for r in primary]
    score = float(np.mean(values)) if all(v is not None for v in values) else 1e100
    return rows, windows, saved, (failure_rate, score)
