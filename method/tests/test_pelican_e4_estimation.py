"""Observation convention, causality, truth isolation and frozen-loop parity."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from pelican_e4_estimation.observation import CausalPolynomialPose, ObservationPipeline
from pelican_e4.environment import noise_samples, observation_history
from pelican_e3_corrected.control import BiasMPC
from run_pelican_e2 import restore_model
from run_pelican_e4 import run_case as original_case
from run_pelican_e4_estimation_diagnostic import run_case


class EstimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src=ROOT/'runs/pelican_e4_v1'
        cls.cfg=json.loads((src/'e3_config.json').read_text());cls.cfg['duration_s']=.2
        cls.params=json.loads((src/'plant_parameters.json').read_text())
        cls.motor=json.loads((src/'motor_model.json').read_text())
        cls.settings=json.loads((src/'config.json').read_text())
        cls.scenario=next(s for s in cls.settings['scenarios'] if s['name']=='pose_noise')
        cls.models={k:restore_model(v) for k,v in json.loads((src/'corrected_models.json').read_text()).items()}
        with np.load(src/'initial_conditions/15301.npz') as a:cls.initial={k:a[k] for k in a.files}

    def test_quadratic_motion_and_backward_increment_convention(self):
        est=CausalPolynomialPose(.01)
        anchor=Rotation.from_euler('xyz',[.2,-.3,.1])
        for k in range(100):
            t=k*.01; p=np.array([.3+2*t+.4*t*t,1-.2*t,.7])
            angle=.2*t+.1*t*t
            q=(anchor*Rotation.from_rotvec([0,0,angle])).as_quat()
            x=est.update(p,q)
            if k>=2:
                assert_allclose(x[:3],p,atol=1e-13)
                assert_allclose(x[3:6],[2+.8*(t-.005),-.2,0],atol=1e-12)
                assert_allclose(Rotation.from_quat(x[6:10]).as_matrix(),Rotation.from_quat(q).as_matrix(),atol=1e-13)
                assert_allclose(x[10:13],[0,0,.2+.2*(t-.005)],atol=1e-12)

    def test_quaternion_sign_and_prefix_causality(self):
        rng=np.random.default_rng(1337)
        pos=rng.normal(0,.01,(100,3));q=Rotation.from_rotvec(rng.normal(0,.01,(100,3))).as_quat()
        def replay(p,r):
            f=CausalPolynomialPose(.01)
            return np.asarray([f.update(a,b) for a,b in zip(p,r)])
        original=replay(pos,q);altered=pos.copy();altered[50:]+=10
        assert_array_equal(original[:50],replay(altered,q)[:50])
        signed=q*np.where(np.arange(100)%2,1,-1)[:,None]
        other=replay(pos,signed)
        assert_allclose(other[:,:6],original[:,:6],atol=1e-12)
        assert_allclose(other[:,10:],original[:,10:],atol=1e-12)
        assert_allclose(Rotation.from_quat(other[:,6:10]).as_matrix(),Rotation.from_quat(original[:,6:10]).as_matrix(),atol=1e-13)

    def test_pose_only_estimator_and_oracle_channel_isolation(self):
        noise=noise_samples(15301,100,self.scenario,self.settings)
        modified=deepcopy(self.initial)
        modified['truth'][:,3:6]=999;modified['truth'][:,10:]=999
        modified['states'][:,3:6]=-999;modified['states'][:,10:]=-999
        a=ObservationPipeline('causal_poly',self.initial,noise,.01)
        b=ObservationPipeline('causal_poly',modified,noise,.01)
        assert_array_equal(a.initial_states,b.initial_states)
        raw,_=observation_history(self.initial,noise,.01)
        oracle=ObservationPipeline('oracle_rates',self.initial,noise,.01).initial_states
        assert_array_equal(oracle[:,:3],raw[:,:3]);assert_array_equal(oracle[:,6:10],raw[:,6:10])
        assert_allclose(oracle[:,3:6],self.initial['states'][:,3:6],atol=1e-12)
        assert_allclose(oracle[:,10:13],self.initial['states'][:,10:13],atol=1e-12)
        self.assertLessEqual(max(max(m['lags'],default=0) for m in self.models.values()),
                             len(self.initial['truth'])-11)

    def test_raw_runner_matches_frozen_all_five_controllers(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);log=lambda *a,**kw:None
            for name in self.settings['controllers']:
                ctrl=None if name=='pd' else BiasMPC(self.models[name],self.motor,self.params,self.cfg)
                args=(self.scenario,name,'figure8',15301,ctrl,self.initial,self.params,self.motor,self.cfg,self.settings)
                original_case(*args,p/(name+'_old'),log,time.monotonic())
                run_case(*args,p/(name+'_new'),log,time.monotonic(),mode='raw')
                with np.load(p/(name+'_old')/'trajectory.npz') as a,np.load(p/(name+'_new')/'trajectory.npz') as b:
                    for key in ['physical','observation','command','residual','bias_estimate','motor_prior']:
                        assert_allclose(a[key],b[key],atol=1e-12,rtol=1e-12,err_msg=name+'/'+key)


if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main()
