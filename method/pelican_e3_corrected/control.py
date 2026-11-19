"""E3 repair, preserving the frozen sparse coefficients and MPC objective.

The six bias coordinates are body specific force [m/s^2] and body angular
acceleration [rad/s^2]. They are added before integrating ALL kinematics.
This is a filtered model-error compensator, not an offset-free MPC theorem.
"""
from copy import deepcopy
import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation
from pelican_e2.data import advance, state_features
from pelican_e2.model import library
from pelican_e3.control import MPC, pack_observation, symbolic_transition as frozen_transition


def calibrate_equilibrium(model, motor, cfg, parameters):
    """Anchor at the known nominal E3 simulator hover, without refitting E2.

    Only y_mu changes. This explicitly uses nominal simulator calibration;
    it is not calibration inferred solely from the original flight logs.
    """
    corrected = deepcopy(model)
    u = np.asarray(parameters['hover_command'])
    x = np.r_[0., 0., 1., np.zeros(6), 1., np.zeros(3)]
    gain, offset = np.asarray(motor['coefficients']).T
    length = max(model['lags'], default=0) + 1
    packed = pack_observation(np.tile(x, (length, 1)), np.tile(u, (length, 1)),
                              gain*u+offset, model, cfg)
    fn, _ = frozen_transition(model, motor, cfg)
    predicted = np.asarray(fn(packed, u)).ravel()
    residual = np.r_[(predicted[3:6]-x[3:6])/cfg['dt'],
                     (predicted[10:13]-x[10:13])/cfg['dt']]
    corrected['y_mu'] = np.asarray(model['y_mu']) - residual
    check, _ = frozen_transition(corrected, motor, cfg)
    fixed_point_error = float(np.max(np.abs(np.asarray(check(packed, u)).ravel()-packed)))
    if fixed_point_error > 1e-10:
        raise ValueError('Corrected packed hover is not a fixed point')
    return corrected, {'removed_acceleration_bias': residual,
                       'hover_command': u, 'hover_state': x,
                       'packed_fixed_point_max_error': fixed_point_error,
                       'source': 'Known nominal E3 simulator equilibrium; no test trajectory fitting'}


def symbolic_transition(model, motor, cfg):
    """Extend the original two-input transition with a constant horizon bias."""
    base, dimension = frozen_transition(model, motor, cfg)
    s, u, d = ca.SX.sym('s', dimension), ca.SX.sym('u', 4), ca.SX.sym('d', 6)
    nominal = base(s, u)
    q, dt = s[6:10], cfg['dt']
    twice = 2*ca.cross(q[:3], d[:3])
    force_world = d[:3]+q[3]*twice+ca.cross(q[:3], twice)
    v = nominal[3:6]+dt*force_world
    omega = nominal[10:13]+dt*d[3:]
    a2 = dt*dt*ca.dot(omega, omega)
    angle = ca.sqrt(a2+1e-24)
    factor = ca.if_else(a2 < 1e-8, .5-a2/48+a2*a2/3840, ca.sin(angle/2)/angle)
    dq = ca.vertcat(dt*factor*omega, ca.cos(angle/2))
    qnext = ca.vertcat(q[3]*dq[:3]+dq[3]*q[:3]+ca.cross(q[:3], dq[:3]),
                      q[3]*dq[3]-ca.dot(q[:3], dq[:3]))
    qnext /= ca.sqrt(ca.dot(qnext, qnext)+1e-24)
    result = ca.vertcat(s[:3]+dt*v, v, qnext, omega, nominal[13:])
    return ca.Function('bias_transition_'+model['name'], [s, u, d], [result]), dimension


def numpy_transition(s, u, d, model, motor, cfg):
    """Independent NumPy feature evaluation, also used for replay auditing."""
    x = np.asarray(s[:13]); lagmax = max(model['lags'], default=0)
    history = np.asarray(s[13:13+13*lagmax]).reshape(lagmax, 13)
    if model['motor']:
        poles = np.asarray(motor['poles']); gain, offset = np.asarray(motor['coefficients']).T
        current_motor = poles*s[-4:]+(1-poles)*(gain*u+offset)
        inp = current_motor
    else:
        inp = u
    z = (np.r_[state_features(x, cfg['gravity']), inp]-model['mu'])/model['scale']
    features = library(z[None], {lag: history[lag-1][None] for lag in model['lags']},
                       model['lags'], model['degree'])[0]
    target = (model['coefficients']@(features/model['feature_scale']))*model['y_scale']+model['y_mu']
    result = [advance(x, target+np.asarray(d), cfg)]
    if lagmax:
        result.append(np.vstack([z, history[:-1]]).ravel())
    if model['motor']:
        result.append(current_motor)
    return np.concatenate(result)


class BiasObserver:
    """Causal residual low-pass filter, updated ONLY after observation k+1.

    Prediction always excludes the online estimate. Thus the residual is an
    absolute estimate of the remaining model error, never double-subtracted.
    No physical simulator state, future reference, or latent motor is accepted.
    """
    def __init__(self, transition, dt, settings):
        self.transition, self.dt = transition, dt
        tau = np.asarray(settings['time_constant_s'], dtype=float)
        self.limit = np.asarray(settings['estimate_limit'], dtype=float)
        self.gate = np.asarray(settings['residual_gate'], dtype=float)
        if any(a.shape != (6,) for a in [tau, self.limit, self.gate]):
            raise ValueError('Observer settings require six components')
        if dt <= 0 or np.any(tau <= 0) or np.any(self.limit <= 0) or np.any(self.gate <= 0):
            raise ValueError('Observer settings must be positive')
        self.alpha = 1-np.exp(-dt/tau)
        self.reset()

    def reset(self):
        self.estimate = np.zeros(6)
        self.updates = 0

    def update(self, packed_before, applied_command, observation_after):
        predicted = np.asarray(self.transition(packed_before, applied_command, np.zeros(6))).ravel()
        residual = np.r_[Rotation.from_quat(packed_before[6:10]).inv().apply(
            (observation_after[3:6]-predicted[3:6])/self.dt),
            (observation_after[10:13]-predicted[10:13])/self.dt]
        accepted = bool(np.isfinite(residual).all() and np.all(np.abs(residual) <= self.gate))
        before = self.estimate.copy()
        candidate = before+self.alpha*(residual-before) if accepted else before
        self.estimate = np.clip(candidate, -self.limit, self.limit)
        clipped = bool(np.any(candidate != self.estimate))
        self.updates += 1
        return residual, self.estimate.copy(), accepted, clipped


class BiasMPC(MPC):
    """Same E3 objective, constraints, solver and fallback acceptance rules.

    Extra six parameters are constant across the prediction horizon. The
    frozen MPC.solve handles warm starts, feasibility checks and statistics.
    """
    def __init__(self, model, motor, parameters, cfg):
        self.model, self.cfg, self.parameters = model, cfg, parameters
        self.transition, self.dimension = symbolic_transition(model, motor, cfg)
        self.hold = round(cfg['control_dt']/cfg['dt'])
        self.blocks = cfg['prediction_steps']//self.hold
        assert self.blocks*self.hold == cfg['prediction_steps']
        self.hover, self.scale = np.asarray(parameters['hover_command']), 30.
        control = ca.MX.sym('control', 4, self.blocks)
        s0 = ca.MX.sym('state', self.dimension)
        ref = ca.MX.sym('reference', 13, self.blocks)
        old, bias = ca.MX.sym('old', 4), ca.MX.sym('bias', 6)
        qpos, qvel = ca.DM(cfg['position_weights']), ca.DM(cfg['velocity_weights'])
        s = s0; cost = 0; constraints = []
        for block in range(self.blocks):
            u = ca.DM(self.hover)+self.scale*control[:, block]
            for _ in range(self.hold):
                s = self.transition(s, u, bias)
            error = s[:13]-ref[:, block]
            attitude = 1-ca.dot(s[6:10], ref[6:10, block])**2
            cost += (ca.dot(qpos, error[:3]**2)+ca.dot(qvel, error[3:6]**2)
                     +cfg['attitude_weight']*attitude+cfg['omega_weight']*ca.sumsqr(error[10:13]))
            delta = u-(old if block == 0 else ca.DM(self.hover)+self.scale*control[:, block-1])
            cost += cfg['input_weight']*ca.sumsqr(control[:, block])+cfg['input_change_weight']*ca.sumsqr(delta/self.scale)
            constraints.append(delta)
        cost += cfg['terminal_factor']*ca.dot(qpos, (s[:3]-ref[:3, -1])**2)
        # Bias follows state so frozen solve can concatenate [augmented, ref, old].
        nlp = {'x': ca.vec(control), 'p': ca.vertcat(s0, bias, ca.vec(ref), old),
               'f': cost, 'g': ca.vertcat(*constraints)}
        opts = {'print_time': False, 'ipopt.print_level': 0, 'ipopt.sb': 'yes',
                'ipopt.max_iter': cfg['solver_max_iter'], 'ipopt.tol': cfg['solver_tolerance'],
                'ipopt.acceptable_tol': cfg['solver_tolerance']*10, 'ipopt.acceptable_iter': 4,
                'ipopt.hessian_approximation': 'limited-memory', 'ipopt.max_cpu_time': cfg['solver_max_seconds']}
        self.solver = ca.nlpsol('bias_mpc_'+model['name'], 'ipopt', nlp, opts)
        self.lbx = np.tile((np.asarray(parameters['command_lower'])-self.hover)/self.scale, self.blocks)
        self.ubx = np.tile((np.asarray(parameters['command_upper'])-self.hover)/self.scale, self.blocks)
        self.warm = np.zeros((4, self.blocks))

    def solve(self, observation, references, previous, bias=None):
        d = np.zeros(6) if bias is None else np.asarray(bias)
        if d.shape != (6,) or not np.isfinite(d).all():
            raise ValueError('Finite six-component horizon bias required')
        return super().solve(np.r_[observation, d], references, previous)
