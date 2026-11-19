"""Check reference derivatives, time scaling and control-only parameter preservation."""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
from numpy.testing import assert_allclose
from pelican_e3.reference import translation
from run_pelican_e3 import initialize
from run_pelican_speed_sweep import case_config, read_json
from verify_pelican_speed_sweep import reference_position_velocity


class SpeedSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / 'runs/pelican_e3_nominal_comparison_v1'
        cls.base = read_json(source / 'e3_config.json')
        cls.params = read_json(source / 'plant_parameters.json')
        cls.motor = read_json(source / 'motor_model.json')

    def test_reference_derivatives_including_smooth_start(self):
        for w in [.45, .675, .9]:
            cfg = case_config(self.base, w, 20.)
            for t in [.3, 1.0, 1.7, 3., 8.]:
                p, v, a = translation(t, 'figure8', cfg)
                pm, vm, _ = translation(t - 1e-5, 'figure8', cfg)
                pp, vp, _ = translation(t + 1e-5, 'figure8', cfg)
                assert_allclose(v, (pp - pm) / 2e-5, atol=1e-8)
                assert_allclose(a, (vp - vm) / 2e-5, atol=1e-8)
                assert_allclose(np.r_[p, v], reference_position_velocity(np.array([t]), cfg)[0], atol=1e-12)

    def test_same_phase_velocity_and_acceleration_scaling(self):
        phase = 2.
        p0, v0, a0 = translation(phase / .45 + 1., 'figure8', self.base)
        for factor in [1., 1.5, 2.]:
            cfg = case_config(self.base, .45 * factor, 20.)
            p, v, a = translation(phase / (.45 * factor) + 1., 'figure8', cfg)
            assert_allclose(p, p0, atol=1e-12)
            assert_allclose(v, factor * v0, atol=1e-12)
            assert_allclose(a, factor**2 * a0, atol=1e-12)

    def test_same_plant_controller_parameters_and_warmup(self):
        baseline = initialize(17301, self.params, self.motor, self.base)
        for w in [.45, .675, .9]:
            cfg = case_config(self.base, w, 20.)
            for key in self.base:
                if key != 'trajectory_frequency_rad_s':
                    self.assertEqual(cfg[key], self.base[key])
            other = initialize(17301, self.params, self.motor, cfg)
            for key in baseline:
                assert_allclose(other[key], baseline[key], atol=0, rtol=0)
        self.assertEqual(self.base['trajectory_frequency_rad_s'], .45)


if __name__ == '__main__':
    unittest.main()
