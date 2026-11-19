"""Causal, pose-only polynomial estimation and explicitly privileged controls.

The pose estimator knows neither the plant, commands, references, nor truth.
It fits a quadratic over the most recent 11 pose samples (100 ms span).
Rates are the fitted pose's backward increment over the last sampling interval,
matching the target convention of the frozen models. Attitude fits use local
SO(3) logarithms; quaternions are never averaged componentwise.
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from pelican_e3.plant import observe
from pelican_e4.environment import measured_pose


class CausalPolynomialPose:
    def __init__(self, dt, window=11, degree=2):
        if dt <= 0 or window < 3 or degree != 2:
            raise ValueError('Positive dt, window >= 3, quadratic fit required')
        self.dt, self.window = float(dt), int(window)
        self.positions, self.quaternions = deque(maxlen=window), deque(maxlen=window)
        self.weights = {}
        for n in range(2, window+1):
            # Dimensionless sample offsets keep the least-squares solve scaled.
            a = np.vander(np.arange(1-n, 1, dtype=float), N=min(degree, n-1)+1, increasing=True)
            inv = np.linalg.pinv(a)
            self.weights[n] = (inv[0], np.power(-1., np.arange(a.shape[1])) @ inv)

    def update(self, position, quaternion):
        p, q = np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
            raise ValueError('Only a finite 3D position and xyzw quaternion are accepted')
        q = Rotation.from_quat(q).as_quat()
        self.positions.append(p.copy()); self.quaternions.append(q.copy())
        if len(self.positions) == 1:
            return np.r_[p, np.zeros(3), q, np.zeros(3)]
        current, previous = self.weights[len(self.positions)]
        positions = np.asarray(self.positions)
        anchor = Rotation.from_quat(q)
        local = (anchor.inv()*Rotation.from_quat(np.asarray(self.quaternions))).as_rotvec()
        r0 = anchor*Rotation.from_rotvec(current@local)
        r1 = anchor*Rotation.from_rotvec(previous@local)
        return np.r_[current@positions, (current-previous)@positions/self.dt,
                     r0.as_quat(), (r1.inv()*r0).as_rotvec()/self.dt]


class ObservationPipeline:
    """Only this diagnostic wrapper can access truth; the estimator cannot.

    raw: exactly frozen E4 backward differences.
    oracle_rates: noisy position/attitude, clean backward velocity/rate ONLY.
    causal_poly: pose-only estimation, same fixed parameters for every method.
    """
    def __init__(self, mode, initial, noise, dt, window=11):
        if mode not in ('raw', 'oracle_rates', 'causal_poly'):
            raise ValueError(mode)
        self.mode, self.dt = mode, dt
        self.filter = CausalPolynomialPose(dt, window)
        states, raw = [], []
        for k, truth in enumerate(initial['truth']):
            pose = measured_pose(truth, noise[k])
            if k == 0:
                original = initial['states'][0].copy()
                original[:3], original[6:10] = pose[:3], pose[6:10]
                clean = initial['states'][0].copy()
            else:
                original = observe(self.previous_pose, pose, dt)
                clean = observe(self.previous_truth, truth, dt)
            states.append(self._choose(pose, original, clean))
            raw.append(original)
            self.previous_pose, self.previous_truth = pose, truth.copy()
        self.initial_states = np.asarray(states)
        self.initial_raw = np.asarray(raw)

    def _choose(self, pose, raw, clean):
        if self.mode == 'raw':
            return raw.copy()
        if self.mode == 'oracle_rates':
            result = raw.copy()
            result[3:6], result[10:13] = clean[3:6], clean[10:13]
            return result
        return self.filter.update(pose[:3], pose[6:10])

    def update(self, truth, noise):
        pose = measured_pose(truth, noise)
        raw = observe(self.previous_pose, pose, self.dt)
        clean = observe(self.previous_truth, truth, self.dt)
        state = self._choose(pose, raw, clean)
        self.previous_pose, self.previous_truth = pose, truth.copy()
        return state, raw, clean
