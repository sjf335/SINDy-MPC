"""Cross-runner equivalence, causal actuator state, and timing-accounting checks."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from run_pelican_e2 import restore_model
from run_pelican_e3 import initialize, run_case as original_case
from run_pelican_e3_corrected import run_case as corrected_case
from run_pelican_e3_nominal_comparison import run_case, full_period_time
from pelican_e3.control import pack_observation
from pelican_e3_corrected.control import BiasMPC, calibrate_equilibrium, numpy_transition


class NominalComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = ROOT/'runs/pelican_e3_v1'
        cls.cfg = json.loads((src/'config.json').read_text()); cls.cfg['duration_s'] = .2
        cls.params = json.loads((src/'plant_parameters.json').read_text())
        cls.motor = json.loads((src/'motor_model.json').read_text())
        cls.settings = json.loads((ROOT/'configs/pelican_e3_nominal_comparison.json').read_text())
        cls.models = {m['name']: restore_model(m) for m in json.loads((src/'selected_models.json').read_text()) if m['name'] in cls.settings['controllers']}
        cls.initial = initialize(12301, cls.params, cls.motor, cls.cfg)

    def test_full_period_includes_five_causal_updates_without_double_counting(self):
        pack = [.001]*11; feedback = [0.]+[.002]*11
        self.assertAlmostEqual(full_period_time(0, 5, .030, pack, feedback), .030)
        self.assertAlmostEqual(full_period_time(5, 5, .030, pack, feedback), .044)
        self.assertAlmostEqual(full_period_time(10, 5, .030, pack, feedback), .044)
        feedback[6] = 999.; pack[6] = 999.
        self.assertAlmostEqual(full_period_time(5, 5, .030, pack, feedback), .044)

    def test_pd_trajectory_matches_original_runner(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); started = time.monotonic(); log = lambda *a, **kw: None
            run_case('pd', 'figure8', 12301, None, self.initial, self.params, self.motor, self.cfg,
                     self.settings, p/'new', log, started)
            original_case('pd', 'figure8', 12301, None, self.initial, self.params, self.motor, self.cfg,
                          p/'old', log, started)
            with np.load(p/'new/trajectory.npz') as a, np.load(p/'old/trajectory.npz') as b:
                assert_allclose(a['physical'], b['physical'], atol=1e-12)
                assert_allclose(a['command'], b['command'], atol=1e-12)

    def test_history_trajectory_matches_corrected_runner(self):
        m, _ = calibrate_equilibrium(self.models['sindy_history'], self.motor, self.cfg, self.params)
        ctrl = BiasMPC(m, self.motor, self.params, self.cfg)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); started = time.monotonic(); log = lambda *a, **kw: None
            run_case('sindy_history', 'figure8', 12301, ctrl, self.initial, self.params, self.motor, self.cfg,
                     self.settings, p/'new', log, started)
            old_settings = {**self.settings, 'disturbance': {'start_s': 8., 'body_acceleration': [0.]*6}}
            corrected_case('combined', 'figure8', 'nominal', 12301, ctrl, self.initial, self.params, self.motor,
                           self.cfg, old_settings, p/'old', log, started)
            with np.load(p/'new/trajectory.npz') as a, np.load(p/'old/trajectory.npz') as b:
                assert_allclose(a['physical'], b['physical'], atol=1e-10)
                assert_allclose(a['bias_estimate'], b['bias_estimate'], atol=1e-10)

    def test_motor_prior_is_command_only_and_residual_replays(self):
        m, _ = calibrate_equilibrium(self.models['motor_sindy'], self.motor, self.cfg, self.params)
        ctrl = BiasMPC(m, self.motor, self.params, self.cfg)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            run_case('motor_sindy', 'figure8', 12301, ctrl, self.initial, self.params, self.motor,
                     self.cfg, self.settings, p/'case', lambda *a, **kw: None, time.monotonic())
            with np.load(p/'case/trajectory.npz') as f: data = {k: f[k] for k in f.files}
            prior = self.initial['motor_prior'].copy(); poles = np.asarray(self.motor['poles'])
            gain, offset = np.asarray(self.motor['coefficients']).T
            for k, u in enumerate(data['command']):
                assert_allclose(prior, data['motor_prior'][k], atol=1e-13)
                packed = np.r_[data['observation'][k], prior]
                predicted = numpy_transition(packed, u, np.zeros(6), m, self.motor, self.cfg)
                y = data['observation'][k+1]
                raw = np.r_[Rotation.from_quat(packed[6:10]).inv().apply((y[3:6]-predicted[3:6])/self.cfg['dt']),
                            (y[10:13]-predicted[10:13])/self.cfg['dt']]
                assert_allclose(raw, data['residual'][k], atol=1e-11)
                prior = poles*prior+(1-poles)*(gain*u+offset)
            assert_allclose(prior, data['motor_prior'][-1], atol=1e-13)


if __name__ == '__main__':
    with threadpool_limits(limits=1): unittest.main()
