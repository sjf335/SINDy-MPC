"""Train the history-augmented SINDy model on the public Pelican flight archive."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from dynamics import prepare_flight
from identification import fit_candidate, scales_from_training, training_statistics, validation_score

SETTINGS = {
    'dt': 0.01, 'gravity': 9.81, 'fit_stride': 2, 'training_start_index': 200,
    'origin_start_index': 200, 'origin_stride': 100, 'horizons_s': [0.01, 0.1, 0.25, 0.5, 1.0],
    'primary_horizon_s': 0.5, 'max_speed_m_s': 50., 'max_body_rate_rad_s': 100.,
    'history_lags': [[1, 2, 5], [1, 2, 5, 10], [1, 2, 5, 10, 20]],
}


def flight_split(count):
    if count != 54:
        raise ValueError('The published split expects the 54-flight Pelican MAT file.')
    order = np.random.default_rng(20261026).permutation(count) + 1
    return {name: sorted(ids.tolist()) for name, ids in
            [('train', order[:32]), ('validation', order[32:43]), ('test', order[43:])]}


def fit_flights(raw_flights):
    cfg = SETTINGS
    split = flight_split(len(raw_flights))
    def prepare(ids):
        return [prepare_flight(raw_flights[i-1], i, cfg['dt']) for i in ids]
    train = prepare(split['train'])
    scaling = scales_from_training(train, cfg)
    gram, rhs, _ = training_statistics(train, cfg, scaling)
    del train
    validation = prepare(split['validation'])
    best, best_score = None, None
    for lags in cfg['history_lags']:
        for alpha in (0.0001, 0.01):
            for threshold in (0.005, 0.02, 0.05):
                model = fit_candidate(gram, rhs, scaling, lags, 2, alpha, threshold, cfg)
                model.update(name='sindy_history', family='sindy_history')
                score = validation_score(model, validation, cfg)
                if best is None or score < best_score:
                    best, best_score = model, score
    return best, split


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('flight_mat', type=Path)
    parser.add_argument('--out', type=Path, default=Path('fitted_history_model.json'))
    args = parser.parse_args()
    flights = loadmat(args.flight_mat, simplify_cells=True, variable_names=['flights'])['flights']
    with threadpool_limits(limits=1):
        fitted, ids = fit_flights(flights)
    def encode(value):
        if isinstance(value, (np.ndarray, np.generic)):
            return value.tolist()
        raise TypeError(type(value).__name__)
    args.out.write_text(json.dumps({'model': fitted, 'split': ids}, default=encode,
                                   indent=2, allow_nan=False), encoding='utf-8')
    print(f'Saved {args.out} with {fitted["feature_count"]} candidate features.')
