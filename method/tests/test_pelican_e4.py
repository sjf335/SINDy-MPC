"""Meaningful environment, causality, and cross-runner equivalence checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from run_pelican_e2 import restore_model
from run_pelican_e3 import initialize
from run_pelican_e3_nominal_comparison import run_case as nominal_case
from run_pelican_e4 import run_case
from pelican_e3_corrected.control import BiasMPC
from pelican_e3.plant import derivative, observe
from pelican_e4.environment import plant_parameters, forcing, physical_step, noise_samples, measured_pose, observation_history, event_metrics


class E4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = ROOT/'runs/pelican_e3_nominal_comparison_v1'
        cls.cfg = json.loads((src/'e3_config.json').read_text()); cls.cfg['duration_s'] = .2
        cls.params = json.loads((src/'plant_parameters.json').read_text())
        cls.motor = json.loads((src/'motor_model.json').read_text())
        cls.settings = json.loads((ROOT/'configs/pelican_e4.json').read_text())
        cls.scenarios = {s['name']: s for s in cls.settings['scenarios']}
        cls.models = {n: restore_model(m) for n,m in json.loads((src/'corrected_models.json').read_text()).items()}
        cls.initial = initialize(14301, cls.params, cls.motor, cls.cfg)

    def test_zero_intervention_matches_all_five_e3_controllers(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); log = lambda *a, **kw: None
            for name in self.settings['controllers']:
                controller = None if name == 'pd' else BiasMPC(self.models[name], self.motor, self.params, self.cfg)
                nominal_case(name,'figure8',14301,controller,self.initial,self.params,self.motor,self.cfg,
                             self.settings,p/(name+'_old'),log,time.monotonic())
                run_case(self.scenarios['nominal'],name,'figure8',14301,controller,self.initial,self.params,
                         self.motor,self.cfg,self.settings,p/(name+'_new'),log,time.monotonic())
                with np.load(p/(name+'_old')/'trajectory.npz') as a, np.load(p/(name+'_new')/'trajectory.npz') as b:
                    for key in ['physical','observation','command','residual','bias_estimate','motor_prior']:
                        assert_allclose(a[key],b[key],atol=1e-12,rtol=1e-12,err_msg=name+'/'+key)

    def test_tau_changes_only_plant_and_matches_analytic_motor_solution(self):
        before = deepcopy(self.params); changed = plant_parameters(self.params,self.scenarios['tau_2'])
        self.assertEqual(self.params,before)
        for key in before:
            if key != 'motor_tau': self.assertEqual(changed[key],before[key])
        x = self.initial['truth'][-1]; u = np.asarray(self.params['hover_command'])+2
        after = physical_step(x,u,.01,changed,.0025,np.zeros(3))
        steady = np.asarray(changed['motor_gain'])*u+changed['motor_bias']
        exact = steady+(x[13:17]-steady)*np.exp(-.01/np.asarray(changed['motor_tau']))
        assert_allclose(after[13:17],exact,atol=1e-8,rtol=1e-10)

    def test_world_force_boundaries_and_independent_adaptive_integration(self):
        s=self.scenarios['force_pulse']; z=np.zeros(3); force=np.array([.4,-.25,.15])
        for t,expected in [(7.99,z),(8.,force),(11.99,force),(12.,z)]:
            assert_array_equal(forcing(t,s,self.settings),expected)
        x=self.initial['truth'][-1]; u=np.asarray(self.params['hover_command'])
        def rhs(t,v):
            result=derivative(v,u,self.params); result[3:6]+=force; return result
        exact=solve_ivp(rhs,[0,.01],x,rtol=1e-12,atol=1e-13).y[:,-1]
        exact[6:10]/=np.linalg.norm(exact[6:10])
        assert_allclose(physical_step(x,u,.01,self.params,.0025,force),exact,atol=2e-8,rtol=2e-8)

    def test_paired_noise_pose_geometry_and_causal_differences(self):
        s=self.scenarios['pose_noise']; noise=noise_samples(14301,100,s,self.settings)
        assert_array_equal(noise,noise_samples(14301,100,s,self.settings))
        self.assertFalse(np.array_equal(noise,noise_samples(14302,100,s,self.settings)))
        truth=self.initial['truth']; states,pose=observation_history(self.initial,noise[:len(truth)],.01)
        for k in range(1,len(truth)):
            q=Rotation.from_quat(truth[k,6:10])*Rotation.from_rotvec(noise[k,3:])
            assert_allclose(states[k,:3],truth[k,:3]+noise[k,:3],atol=1e-14)
            assert_allclose(Rotation.from_quat(states[k,6:10]).as_matrix(),q.as_matrix(),atol=1e-14)
            assert_allclose(states[k,3:6],(truth[k,:3]+noise[k,:3]-truth[k-1,:3]-noise[k-1,:3])/.01,atol=1e-12)
        altered=noise.copy(); altered[len(truth):]=100
        earlier,_=observation_history(self.initial,altered[:len(truth)],.01)
        assert_array_equal(states,earlier)
        assert max(max(m['lags'],default=0) for m in self.models.values()) < len(truth)-1

    def test_recovery_requires_contiguous_dwell_and_retains_unrecovered(self):
        t=np.arange(2001)*.01; errors=np.full(len(t),.1); errors[t>=13.2]=.01
        m=event_metrics(t,errors,True,self.settings)
        self.assertTrue(m['recovered']); self.assertAlmostEqual(m['recovery_time_s'],1.2)
        short=event_metrics(t[:1300],errors[:1300],False,self.settings)
        self.assertFalse(short['recovered']); self.assertIsNone(short['recovery_time_s'])
        self.assertIsNone(short['post_event_rmse_m'])


if __name__=='__main__':
    with threadpool_limits(limits=1): unittest.main()
