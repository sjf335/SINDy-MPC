"""Verify all reference families and the worst-baseline/late-window selection rule."""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pelican_e3.reference import translation
from pelican_trajectory_search_common import read_json, grid, configuration, rank, refinement, ORDER, candidate
from verify_pelican_trajectory_search import reference_position_velocity


class TrajectorySearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = read_json(ROOT / 'configs/pelican_trajectory_search.json')
        cls.base = read_json(ROOT / 'runs/pelican_e3_nominal_comparison_v1/e3_config.json')

    def test_all_candidate_reference_derivatives(self):
        for c in grid(self.settings):
            cfg = configuration(self.base, c, 20.)
            for t in [.3, 1.4, 3., 8.]:
                p, v, a = translation(t, c['trajectory'], cfg)
                pm, vm, _ = translation(t - 1e-5, c['trajectory'], cfg)
                pp, vp, _ = translation(t + 1e-5, c['trajectory'], cfg)
                assert_allclose(v, (pp - pm) / 2e-5, atol=1e-8)
                assert_allclose(a, (vp - vm) / 2e-5, atol=1e-8)
                assert_allclose(np.r_[p, v], reference_position_velocity(np.array([t]), cfg, c['trajectory'])[0], atol=1e-12)

    def test_score_penalizes_losing_to_pd_or_in_late_window(self):
        rows = []
        for index in range(3):
            c = candidate('circle', .45 + index * .1, .7)
            for name in ORDER:
                full, late = (1., 1.) if name == 'sindy_history' else (2., 2.)
                if index == 0 and name == 'pd':
                    full = .8
                if index == 1 and name == 'linear_history':
                    late = .9
                rows.append(dict(**c, controller=name, seed=1, position_rmse_m=full, late_rmse_m=late,
                    complete=True, tracking_success=True, fallbacks=0, input_bound_violations=0, input_slew_violations=0))
        result = rank(pd.DataFrame(rows))
        self.assertAlmostEqual(result.iloc[0].frequency_rad_s, .65)
        self.assertAlmostEqual(result.iloc[0].score_percent, 50.)
        self.assertLess(result.iloc[1].score_percent, 0)
        frame = pd.DataFrame(rows)
        frame.loc[frame.frequency_rad_s == .65, 'complete'] = False
        result = rank(frame)
        self.assertFalse(bool(result.iloc[-1].eligible))

    def test_grid_refinement_and_holdout_separation(self):
        points = grid(self.settings)
        self.assertEqual(len(points), 60)
        self.assertEqual(len({c['candidate_id'] for c in points}), 60)
        winners = [points[0], next(c for c in points if c['trajectory'] == 'figure8'),
                   next(c for c in points if c['trajectory'] == 'helix')]
        refined = refinement(winners, points, self.settings)
        self.assertFalse({c['candidate_id'] for c in points} & {c['candidate_id'] for c in refined})
        for c in refined:
            self.assertGreaterEqual(c['frequency_rad_s'], .30)
            self.assertLessEqual(c['frequency_rad_s'], 1.5)
            self.assertGreaterEqual(c['climb_m_s'], .03)
        development = set(self.settings['development_seeds'] + self.settings['confirmation_seeds'])
        self.assertFalse(development & set(self.settings['validation_seeds']))


if __name__ == '__main__':
    unittest.main()
