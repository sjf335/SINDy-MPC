"""Causality, response recovery and flight isolation checks for Pelican E1."""
from pathlib import Path
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_pelican_e1 import (response_features, fit_first_order, predict,
                                    history_design, make_splits, train_scales, fit_command_models)


def response_episode(seed, pole=.8, delay=3, n=700):
    rng = np.random.default_rng(seed)
    u = rng.uniform(20, 150, (n, 4))
    y = np.zeros((n, 4))
    gain, bias = np.array([.4, .5, .6, .7]), np.array([5., 4., 3., 2.])
    for k in range(1, n):
        y[k] = pole*y[k-1]+(1-pole)*(gain*u[max(0, k-delay)]+bias)
    return {'id': seed, 'split': 'train', 'u': u, 'y': y}, gain, bias


class PelicanE1Tests(unittest.TestCase):
    def setUp(self):
        self.config = {'prediction_start': 20, 'warmup_samples': 20,
                       'fir_history_samples': [3], 'ridge_alphas': [1e-10]}

    def test_first_order_gain_and_offset_recovered(self):
        a, gain, bias = response_episode(1)
        b, _, _ = response_episode(2)
        coef, mse = fit_first_order([a, b], .8, 3, self.config)
        assert_allclose(coef[:, 0], gain, atol=1e-8)
        assert_allclose(coef[:, 1], bias, atol=1e-7)
        model = {'kind': 'first_order', 'poles': np.full(4, .8), 'delay': 3, 'coefficients': coef}
        c, _, _ = response_episode(3)
        assert_allclose(predict(model, c, self.config), c['y'][20:], atol=1e-8)

    def test_no_future_command_or_reported_speed_leakage(self):
        e, gain, bias = response_episode(5)
        model = {'kind': 'first_order', 'poles': np.full(4, .8), 'delay': 3,
                 'coefficients': np.column_stack([gain, bias])}
        before = predict(model, e, self.config)
        changed = {**e, 'u': e['u'].copy(), 'y': e['y'].copy()}
        changed['y'][20:] += 10000
        assert_allclose(predict(model, changed, self.config), before)
        changed['u'][101:] -= 200
        after = predict(model, changed, self.config)
        assert_allclose(after[:81], before[:81])  # original timestamps 20..100

    def test_history_orientation_and_no_cross_flight_samples(self):
        e, _, _ = response_episode(8)
        x = history_design(e['u'], 3, 20, 0, np.zeros(4), np.ones(4))
        assert_allclose(x[0], np.r_[1, e['u'][[20, 19, 18], 0]])
        with self.assertRaises(ValueError):
            history_design(e['u'], 5, 3, 0, np.zeros(4), np.ones(4))
        train = []
        for seed in [11, 12]:
            f, _, _ = response_episode(seed)
            u = f['u']
            f['y'] = 1+2*u+.5*np.roll(u, 1, axis=0)-.3*np.roll(u, 2, axis=0)
            train.append(f)
        mu, scale = train_scales(train, 40)
        fir = fit_command_models(train, self.config, mu, scale)[-1]
        f, _, _ = response_episode(13)
        u = f['u']
        target = 1+2*u+.5*np.roll(u, 1, axis=0)-.3*np.roll(u, 2, axis=0)
        assert_allclose(predict(fir, f, self.config), target[20:], atol=1e-6)

    def test_split_disjointness_and_invalid_delay(self):
        s = make_splits(54, {'split_seed': 20261026, 'train_flights': 32, 'validation_flights': 11})
        a, b, c = [set(s[k]) for k in ['train', 'validation', 'test']]
        self.assertEqual((len(a), len(b), len(c)), (32, 11, 11))
        self.assertFalse(a & b or a & c or b & c)
        self.assertEqual(a | b | c, set(range(1, 55)))
        with self.assertRaises(ValueError):
            response_features(np.zeros((100, 4)), np.zeros(4), .8, -1, 20)


if __name__ == '__main__':
    unittest.main()
