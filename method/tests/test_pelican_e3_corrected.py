"""Checks of equilibrium, bias sign/frame/causality and frozen MPC equivalence."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from pelican_e3.control import MPC, pack_observation, symbolic_transition as old_transition
from pelican_e3.plant import derivative, rotate
from pelican_e3.reference import at
from pelican_e3_corrected.control import BiasMPC, BiasObserver, calibrate_equilibrium, symbolic_transition, numpy_transition
from pelican_e3_corrected.plant import disturbed_step
from run_pelican_e2 import restore_model
from run_pelican_e3 import initialize


class CorrectedE3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = ROOT/'runs/pelican_e3_v1'
        cls.cfg = json.loads((src/'config.json').read_text())
        cls.p = json.loads((src/'plant_parameters.json').read_text())
        cls.motor = json.loads((src/'motor_model.json').read_text())
        cls.models = [restore_model(m) for m in json.loads((src/'selected_models.json').read_text()) if m['name'] in cls.cfg['controllers']]
        cls.model = next(m for m in cls.models if m['name'] == 'sindy_history')
        cls.settings = json.loads((ROOT/'configs/pelican_e3_corrected.json').read_text())['observer']
        cls.initial = initialize(10301, cls.p, cls.motor, cls.cfg)

    def packed(self, model):
        a = self.initial
        return pack_observation(a['states'], a['commands'], a['motor_prior'], model, self.cfg)

    def test_equilibrium_full_history_and_original_preserved(self):
        for m in self.models:
            before = m['y_mu'].copy()
            corrected, meta = calibrate_equilibrium(m, self.motor, self.cfg, self.p)
            assert_array_equal(m['y_mu'], before)
            assert_array_equal(m['coefficients'], corrected['coefficients'])
            self.assertLess(meta['packed_fixed_point_max_error'], 1e-11)
            assert_allclose(corrected['y_mu']+meta['removed_acceleration_bias'], before, atol=1e-14)
            length = max(m['lags'], default=0)+1
            gain, offset = np.asarray(self.motor['coefficients']).T
            u = np.asarray(meta['hover_command']); x = meta['hover_state']
            s = pack_observation(np.tile(x, (length, 1)), np.tile(u, (length, 1)), gain*u+offset, m, self.cfg)
            f, _ = symbolic_transition(corrected, self.motor, self.cfg)
            for _ in range(50):
                s = np.asarray(f(s, u, np.zeros(6))).ravel()
            assert_allclose(s[:13], x, atol=1e-10)

    def test_zero_bias_matches_frozen_and_nonzero_numpy_multistep(self):
        rng = np.random.default_rng(714)
        for m in self.models:
            f, _ = symbolic_transition(m, self.motor, self.cfg)
            old, _ = old_transition(m, self.motor, self.cfg)
            s = self.packed(m); sn = s.copy()
            for _ in range(30):
                u = np.asarray(self.p['hover_command'])+rng.normal(0, .5, 4)
                d = rng.normal(0, .07, 6)
                assert_allclose(np.asarray(f(s, u, np.zeros(6))).ravel(), np.asarray(old(s, u)).ravel(), atol=2e-12)
                s = np.asarray(f(s, u, d)).ravel()
                sn = numpy_transition(sn, u, d, m, self.motor, self.cfg)
                assert_allclose(s, sn, atol=3e-11, rtol=3e-11)

    def test_observer_convergence_rotated_frame_and_no_double_subtraction(self):
        m = self.model; f, _ = symbolic_transition(m, self.motor, self.cfg)
        observer = BiasObserver(f, self.cfg['dt'], self.settings)
        s = self.packed(m); s[6:10] = Rotation.from_euler('xyz', [.3, -.2, .6]).as_quat()
        u = np.asarray(self.p['hover_command'])
        d = np.array([.18, -.13, .09, -.04, .05, .03])
        y = np.asarray(f(s, u, d)).ravel()[:13]
        for _ in range(800):
            raw, estimated, accepted, clipped = observer.update(s, u, y)
            assert_allclose(raw, d, atol=2e-12)
            self.assertTrue(accepted); self.assertFalse(clipped)
        assert_allclose(estimated, d, atol=3e-8)
        observer.reset(); assert_array_equal(observer.estimate, np.zeros(6)); self.assertEqual(observer.updates, 0)

    def test_observer_uses_actual_command_and_rejects_invalid_observation(self):
        m = self.model; f, _ = symbolic_transition(m, self.motor, self.cfg)
        observer = BiasObserver(f, self.cfg['dt'], self.settings)
        s = self.packed(m); u = np.asarray(self.p['hover_command'])+np.array([2., -1., 1., -.5])
        y = np.asarray(f(s, u, np.zeros(6))).ravel()[:13]
        raw, _, accepted, _ = observer.update(s, u, y)
        assert_allclose(raw, 0, atol=1e-12); self.assertTrue(accepted)
        assert_array_equal(observer.estimate, np.zeros(6))
        wrong, _, _, _ = observer.update(s, u+2., y)
        self.assertGreater(np.linalg.norm(wrong), .01)
        before = observer.estimate.copy(); y[3] = np.nan
        _, after, accepted, _ = observer.update(s, u, y)
        self.assertFalse(accepted); assert_array_equal(before, after)
        # Explicit estimate projection and innovation gate are separate.
        observer.reset(); y = np.asarray(f(s, u, np.ones(6)*10)).ravel()[:13]
        for _ in range(300):
            _, _, accepted, clipped = observer.update(s, u, y)
        self.assertTrue(accepted); self.assertTrue(clipped)
        assert_allclose(observer.estimate, self.settings['estimate_limit'])

    def test_zero_bias_mpc_objective_constraints_and_command_match(self):
        m = self.model
        old = MPC(m, self.motor, self.p, self.cfg); new = BiasMPC(m, self.motor, self.p, self.cfg)
        s = self.packed(m); prior = self.initial['commands'][-1]
        refs = np.array([at(k*self.cfg['control_dt'], 'figure8', self.cfg)[0] for k in range(1, old.blocks+1)])
        u0, a = old.solve(s, refs, prior); u1, b = new.solve(s, refs, prior, np.zeros(6))
        self.assertTrue(a['success'] and b['success'])
        assert_allclose(u0, u1, atol=2e-6, rtol=0)
        assert_allclose(a['objective'], b['objective'], atol=2e-8)

    def test_disturbance_integration_against_adaptive_solver(self):
        x = self.initial['truth'][-1]; u = np.asarray(self.p['hover_command']); d = np.array([.2, -.1, .12, .03, -.02, .01])
        def rhs(t, s):
            result = derivative(s, u, self.p)
            result[3:6] += rotate(s[6:10], d[:3]); result[10:13] += d[3:]
            return result
        expected = solve_ivp(rhs, [0, .01], x, rtol=1e-12, atol=1e-13).y[:, -1]
        assert_allclose(disturbed_step(x, u, .01, self.p, .0025, d), expected, atol=2e-8, rtol=2e-8)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
