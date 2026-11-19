"""Fit and validate a sparse body-acceleration predictor with motion and input history."""
import warnings
import numpy as np
from pysindy import STLSQ
from dynamics import advance, state_features, target_acceleration

PAIR_I, PAIR_J = np.triu_indices(13)
MEMORY_COLUMNS = np.r_[np.arange(6), np.arange(9, 13)]
CURRENT_SIZE = 1 + 13 + len(PAIR_I)


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


def scales_from_training(train, config):
    means, seconds, ymeans, yseconds = [], [], [], []
    for e in train:
        idx = np.arange(config['training_start_index'], len(e['x'])-1, config['fit_stride'])
        z = np.c_[state_features(e['x'][idx], config['gravity']),
                  e['u'][idx]]
        y = target_acceleration(e, idx, config)
        means.append(z.mean(0)); seconds.append((z*z).mean(0))
        ymeans.append(y.mean(0)); yseconds.append((y*y).mean(0))
    mu, ymu = np.mean(means, 0), np.mean(ymeans, 0)
    scale = np.sqrt(np.maximum(np.mean(seconds, 0)-mu*mu, 1e-6))
    yscale = np.sqrt(np.maximum(np.mean(yseconds, 0)-ymu*ymu, 1e-6))
    return {'mu': mu, 'scale': scale, 'y_mu': ymu, 'y_scale': yscale}


def training_statistics(train, config, scaling):
    maxlags = config['history_lags'][-1]
    size = CURRENT_SIZE+14*len(maxlags)
    gram, rhs = np.zeros((size, size)), np.zeros((size, 6))
    samples = 0
    for e in train:
        zall = np.c_[state_features(e['x'], config['gravity']), e['u']]
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


def fit_candidate(gram, rhs, scaling, lags, degree, alpha, threshold, config):
    maxlags = config['history_lags'][-1]
    idx = feature_indices(lags, degree, maxlags)
    g, c = gram[np.ix_(idx, idx)], rhs[idx]
    feature_scale = np.sqrt(np.maximum(np.diag(g), 1e-12))
    gn = g/feature_scale[:, None]/feature_scale[None, :]
    cn = c/feature_scale[:, None]
    x, y, rank = compressed_design(gn, cn)
    estimator = STLSQ(threshold=threshold, alpha=alpha, normalize_columns=False,
                      unbias=False, max_iter=40)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        estimator.fit(x, y)
    warning_messages = [str(w.message) for w in captured]
    coefficient = np.asarray(estimator.coef_)
    return {**scaling, 'lags': list(lags), 'degree': degree, 'alpha': alpha, 'threshold': threshold,
            'motor': False, 'coefficients': coefficient, 'feature_scale': feature_scale,
            'rank': rank, 'nonzero_coefficients': int(np.count_nonzero(coefficient)),
            'feature_count': len(feature_scale), 'fit_warnings': warning_messages}


def context_for_episode(episode, origins, model, config):
    lags = model.get('lags', [])
    length = max(lags, default=0)+1
    indices = origins[:, None]-np.arange(length)[None, :]
    if indices.min() < 0:
        raise ValueError('History would cross a flight boundary')
    xhist = episode['x'][indices]
    input_hist = episode['u'][indices]
    raw = np.concatenate([state_features(xhist, config['gravity']), input_hist], axis=-1)
    history = (raw-np.asarray(model['mu']))/np.asarray(model['scale'])
    return episode['x'][origins].copy(), history.copy()


def rollout(initial, history, input_sequence, model, config, keep_all=False):
    x = initial.copy()
    history = history.copy()
    steps = input_sequence.shape[1]
    alive = np.ones(len(x), dtype=bool)
    failed_at = np.full(len(x), steps+1, dtype=int)
    wanted = {round(t/config['dt']) for t in config['horizons_s']}
    predictions = {}
    all_states = [x.copy()] if keep_all else None
    for j in range(steps):
        u = input_sequence[:, j]
        z = (np.c_[state_features(x, config['gravity']), u]-model['mu'])/model['scale']
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


def validation_score(model, flights, config):
    horizon = round(config['primary_horizon_s'] / config['dt'])
    failure_rates, velocity_errors = [], []
    for flight in flights:
        origins = origins_for(flight, config)
        initial, history = context_for_episode(flight, origins, model, config)
        commands = flight['u'][origins[:, None] + np.arange(horizon)]
        states, failed_at, _ = rollout(initial, history, commands, model, config)
        completed = failed_at > horizon
        failure_rates.append(1 - completed.mean())
        if completed.any():
            delta = states[horizon][completed, 3:6] - flight['x'][origins[completed] + horizon, 3:6]
            velocity_errors.append(np.sqrt(np.mean(np.sum(delta**2, axis=1))))
        else:
            velocity_errors.append(float('inf'))
    return np.mean(failure_rates), np.mean(velocity_errors), model['nonzero_coefficients']
