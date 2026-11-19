"""Scientific correctness checks for pose-derived states and free-running prediction."""
from pathlib import Path
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from pysindy import STLSQ

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pelican_e2.data import (prepare_flight, state_features, target_acceleration, advance,
                             rotate, qmultiply, qexp, motor_estimate)
from pelican_e2.model import compressed_design, context_for_episode, rollout, library


MOTOR = {'poles': [.8]*4, 'coefficients': [[.5, 3.]]*4}
CFG = {'dt': .01, 'gravity': 9.81, 'horizons_s': [.01,.1],
       'max_speed_m_s': 50., 'max_body_rate_rad_s': 100.}


def raw_flight(n=100):
    t = np.arange(n)*.01
    position = np.c_[t, .2*t*t, 1+.1*np.sin(t)]
    euler = np.c_[.1*np.sin(t), .2*np.sin(.5*t), .3*t]
    return {'Pos': position, 'Euler': euler, 'Motors_CMD': np.tile([100.,101.,102.,103.],(n,1))}


class PelicanE2Tests(unittest.TestCase):
    def test_preparation_has_no_added_future_state_dependency(self):
        raw = raw_flight()
        e = prepare_flight(raw, 1, .01, MOTOR)
        changed = {key: value.copy() for key,value in raw.items()}
        for value in changed.values(): value[61:] += 20
        other = prepare_flight(changed, 1, .01, MOTOR)
        assert_allclose(e['x'][:60], other['x'][:60], atol=1e-12)
        assert_allclose(e['motor_hat'][:60], other['motor_hat'][:60], atol=1e-12)

    def test_kinematic_and_quaternion_conventions(self):
        e = prepare_flight(raw_flight(), 1, .01, MOTOR)
        idx = np.arange(len(e['x'])-1)
        target = target_acceleration(e, idx, CFG)
        actual = advance(e['x'][idx], target, CFG)
        assert_allclose(actual, e['x'][idx+1], atol=1e-11)
        rng = np.random.default_rng(8)
        r = Rotation.random(40, random_state=rng)
        v = rng.normal(size=(40,3))
        assert_allclose(rotate(r.as_quat(),v), r.apply(v), atol=1e-12)
        assert_allclose(rotate(r.as_quat(),v,inverse=True), r.inv().apply(v), atol=1e-12)
        w = rng.normal(size=(40,3))*.1
        assert_allclose(qmultiply(r.as_quat(),qexp(w)),(r*Rotation.from_rotvec(w)).as_quat(),atol=1e-12)

    def test_compressed_regression_matches_direct_sparse_fit(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(400, 12))/20
        beta = np.zeros((12, 3)); beta[[1,4,8]] = [[1,.5,-.3],[.2,0,.7],[0,-.4,0]]
        y = x@beta+rng.normal(0, .001, (400,3))
        compressed_x, compressed_y, _ = compressed_design(x.T@x, x.T@y)
        a = STLSQ(threshold=.03,alpha=.001,unbias=False).fit(x,y)
        b = STLSQ(threshold=.03,alpha=.001,unbias=False).fit(compressed_x,compressed_y)
        assert_allclose(a.coef_,b.coef_,atol=1e-10)

    def test_constant_twist_rollout_and_no_future_state_access(self):
        e = prepare_flight(raw_flight(),1,.01,MOTOR)
        model = {'name':'constant_twist','family':'constant_twist','lags':[],'motor':False}
        origins = np.array([40])
        x,h,m = context_for_episode(e,origins,model,CFG)
        inputs = e['u'][40:50][None]
        pred, failed, _ = rollout(x,h,inputs,m,model,MOTOR,CFG)
        assert_allclose(pred[10][0,:3],x[0,:3]+.1*x[0,3:6],atol=1e-12)
        e['x'][41:] += 10000
        x2,h2,m2 = context_for_episode(e,origins,model,CFG)
        pred2,_,_ = rollout(x2,h2,inputs,m2,model,MOTOR,CFG)
        assert_allclose(pred2[10],pred[10],atol=1e-12)
        self.assertEqual(failed[0],11)

    def test_history_uses_only_selected_past_features(self):
        z = np.arange(26,dtype=float).reshape(2,13)
        past = {2:z+2,5:z+5}
        phi = library(z,past,[2,5],1)
        self.assertEqual(phi.shape,(2,34))
        assert_allclose(phi[:,14:24],2)
        assert_allclose(phi[:,24:34],5)


if __name__ == '__main__':
    unittest.main()
