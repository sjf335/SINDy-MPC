"""Numerical checks before closed-loop evaluation."""
from pathlib import Path
import sys
import json
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from pelican_e3.plant import derivative,step,observe
from pelican_e3.control import symbolic_transition,pack_observation,bounded_command,pd_command
from pelican_e3.reference import at,translation
from pelican_e2.model import context_for_episode,rollout
from run_pelican_e2 import restore_model
from verify_pelican_e2 import load_episode


class PelicanE3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((ROOT/'configs/pelican_e3.json').read_text())
        cls.params = json.loads((ROOT/'runs/pelican_e3_calibration_aero/plant_parameters.json').read_text())
        cls.motor = json.loads((ROOT/'runs/pelican_e2_v1/motor_model.json').read_text())
        cls.models = [restore_model(m) for m in json.loads((ROOT/'runs/pelican_e2_v1/models/selected_models.json').read_text())
                      if m['name'] in cls.cfg['controllers']]

    def test_hover_and_pd_equilibrium(self):
        p = self.params; u = np.array(p['hover_command'])
        x = np.r_[0.,0.,1.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,np.array(p['motor_gain'])*u+p['motor_bias']]
        assert_allclose(derivative(x,u,p),0,atol=1e-12)
        assert_allclose(step(x,u,.01,p),x,atol=1e-12)
        ref,a = at(0,'hover',self.cfg)
        assert_allclose(pd_command(x[:13],ref,a,p,self.cfg),u,atol=1e-10)

    def test_rk4_against_adaptive_integrator(self):
        p=self.params; u=np.array(p['hover_command'])+np.array([3,-2,1,-1])
        x=np.r_[0.,0.,1.,.2,-.1,.03,0.,0.,0.,1.,.1,-.2,.3,np.array(p['motor_gain'])*p['hover_command']+p['motor_bias']]
        expected=solve_ivp(lambda t,s:derivative(s,u,p),[0,.01],x,rtol=1e-11,atol=1e-12).y[:,-1]
        assert_allclose(step(x,u,.01,p),expected,atol=2e-8,rtol=2e-8)

    def test_symbolic_transition_and_history_match_frozen_numpy(self):
        e=load_episode(ROOT/'runs/pelican_e2_v1',2); origin=400
        cfg={**self.cfg,'horizons_s':[.01,.1],'max_speed_m_s':50.,'max_body_rate_rad_s':100.}
        for m in self.models:
            x,h,prior=context_for_episode(e,np.array([origin]),m,cfg)
            inputs=e['u'][origin:origin+10][None]
            expected,_,_=rollout(x,h,inputs,prior,m,self.motor,cfg)
            packed=pack_observation(e['x'][:origin+1],e['u'][:origin],prior[0],m,cfg)
            f,dimension=symbolic_transition(m,self.motor,cfg)
            self.assertEqual(len(packed),dimension)
            for k in range(10):
                packed=np.asarray(f(packed,inputs[0,k])).ravel()
                if k==0: assert_allclose(packed[:13],expected[1][0],atol=1e-10)
            assert_allclose(packed[:13],expected[10][0],atol=1e-9)

    def test_reference_derivatives_and_shared_bounds(self):
        eps=1e-5
        for kind in ['circle','figure8','helix']:
            for t in [.5,1.5,3.,10.]:
                p,v,a=translation(t,kind,self.cfg)
                before=translation(t-eps,kind,self.cfg); after=translation(t+eps,kind,self.cfg)
                assert_allclose((after[0]-before[0])/(2*eps),v,atol=1e-8)
                assert_allclose((after[1]-before[1])/(2*eps),a,atol=1e-8)
            assert_allclose(translation(0,kind,self.cfg)[1:],0,atol=1e-12)
        previous=np.array(self.params['hover_command'])
        command=bounded_command(np.array([1e3,-1e3,1e3,-1e3]),previous,self.params,self.cfg)
        self.assertLessEqual(np.max(np.abs(command-previous)),self.cfg['command_delta_per_control']+1e-10)


if __name__=='__main__': unittest.main()
