"""Frozen Pelican E2 dynamics exported to CasADi with exact 100 Hz history."""
import time
import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation
from pelican_e2.data import state_features
from .plant import steady_command
from .reference import rotation_from_force


def symbolic_transition(model,motor,cfg):
    lagmax = max(model['lags'],default=0)
    dimension = 13+13*lagmax+(4 if model['motor'] else 0)
    s,u = ca.SX.sym('s',dimension),ca.SX.sym('u',4)
    x = s[:13]
    q = x[6:10]
    def rot(v,inverse=False):
        xyz = -q[:3] if inverse else q[:3]
        twice = 2*ca.cross(xyz,v)
        return v+q[3]*twice+ca.cross(xyz,twice)
    if model['motor']:
        a = ca.DM(motor['poles']); gain,bias = np.asarray(motor['coefficients']).T
        current_motor = a*s[-4:]+(1-a)*(ca.DM(gain)*u+ca.DM(bias))
        inp = current_motor
    else: inp = u
    raw = ca.vertcat(rot(x[3:6],True),x[10:13],rot(ca.DM([0,0,-cfg['gravity']]),True),inp)
    z = (raw-ca.DM(model['mu']))/ca.DM(model['scale'])
    phi = [ca.SX(1),z]
    if model['degree']==2:
        phi += [z[i]*z[j] for i in range(13) for j in range(i,13)]
    history = ca.reshape(s[13:13+13*lagmax],13,lagmax) if lagmax else None
    for lag in model['lags']:
        prev = history[:,lag-1]
        phi += [prev[i]-z[i] for i in [0,1,2,3,4,5,9,10,11,12]]
        if model['degree']==2: phi += [prev[i]**2-z[i]**2 for i in range(9,13)]
    phi = ca.vertcat(*phi)
    coefficients = ca.sparsify(ca.DM(model['coefficients']))
    target = (coefficients@(phi/ca.DM(model['feature_scale'])))*ca.DM(model['y_scale'])+ca.DM(model['y_mu'])
    dt = cfg['dt']
    vel = x[3:6]+dt*(rot(target[:3])+ca.DM([0,0,-cfg['gravity']]))
    omega = x[10:13]+dt*target[3:]
    a2 = dt*dt*ca.dot(omega,omega)
    angle = ca.sqrt(a2+1e-24)
    factor = ca.if_else(a2<1e-8,.5-a2/48+a2*a2/3840,ca.sin(angle/2)/angle)
    dq = ca.vertcat(dt*factor*omega,ca.cos(angle/2))
    qnext = ca.vertcat(q[3]*dq[:3]+dq[3]*q[:3]+ca.cross(q[:3],dq[:3]),q[3]*dq[3]-ca.dot(q[:3],dq[:3]))
    qnext /= ca.sqrt(ca.dot(qnext,qnext)+1e-24)
    result = [x[:3]+dt*vel,vel,qnext,omega]
    if lagmax: result.append(ca.vec(ca.horzcat(z,history[:,:-1])))
    if model['motor']: result.append(current_motor)
    return ca.Function('transition_'+model['name'],[s,u],[ca.vertcat(*result)]),dimension


def pack_observation(states,commands,motor_prior,model,cfg):
    lagmax = max(model['lags'],default=0)
    result = [np.asarray(states[-1])]
    if lagmax:
        previous_x = np.asarray([states[-1-lag] for lag in range(1,lagmax+1)])
        previous_u = np.asarray([commands[-lag] for lag in range(1,lagmax+1)])
        raw = np.c_[state_features(previous_x,cfg['gravity']),previous_u]
        z = (raw-model['mu'])/model['scale']
        result.append(z.ravel())
    if model['motor']: result.append(motor_prior)
    return np.concatenate(result)


def pd_command(x,reference,acceleration,parameters,cfg):
    r = Rotation.from_quat(x[6:10]).as_matrix()
    vb = r.T@x[3:6]
    desired = (acceleration+np.asarray(cfg['pd_position_gain'])*(reference[:3]-x[:3])
               +np.asarray(cfg['pd_velocity_gain'])*(reference[3:6]-x[3:6]))
    force = desired+np.array([0,0,cfg['gravity']])+r@(np.asarray(parameters['linear_drag'])*vb)
    desired_r = rotation_from_force(force).as_matrix()
    skew = .5*(desired_r.T@r-r.T@desired_r)
    error = np.array([skew[2,1],skew[0,2],skew[1,0]])
    omega = x[10:13]
    inertia = np.asarray(parameters['inertia_ratio'])
    torque_target = (-np.asarray(cfg['pd_attitude_gain'])*error
                     -np.asarray(cfg['pd_rate_gain'])*(omega-r.T@desired_r@reference[10:13])
                     +np.cross(omega,inertia*omega)/inertia+np.asarray(parameters['angular_drag'])*omega
                     -np.asarray(parameters.get('aero_moment_matrix',np.zeros((3,3))))@vb)
    collective = max(.5,float(force@r[:,2]))
    return steady_command(np.r_[collective,torque_target],parameters)


def bounded_command(u,previous,parameters,cfg):
    rate = cfg['command_delta_per_control']
    return np.clip(np.clip(u,previous-rate,previous+rate),parameters['command_lower'],parameters['command_upper'])


class MPC:
    def __init__(self,model,motor,parameters,cfg):
        self.model,self.cfg,self.parameters = model,cfg,parameters
        self.transition,self.dimension = symbolic_transition(model,motor,cfg)
        self.hold = round(cfg['control_dt']/cfg['dt'])
        self.blocks = cfg['prediction_steps']//self.hold
        assert self.blocks*self.hold==cfg['prediction_steps']
        self.hover = np.asarray(parameters['hover_command'])
        self.scale = 30.
        control = ca.MX.sym('control',4,self.blocks)
        s0 = ca.MX.sym('state',self.dimension)
        ref = ca.MX.sym('reference',13,self.blocks)
        old = ca.MX.sym('old',4)
        qpos,qvel = ca.DM(cfg['position_weights']),ca.DM(cfg['velocity_weights'])
        s = s0; cost = 0; constraints = []
        for block in range(self.blocks):
            u = ca.DM(self.hover)+self.scale*control[:,block]
            for _ in range(self.hold): s = self.transition(s,u)
            error = s[:13]-ref[:,block]
            attitude = 1-ca.dot(s[6:10],ref[6:10,block])**2
            cost += (ca.dot(qpos,error[:3]**2)+ca.dot(qvel,error[3:6]**2)
                     +cfg['attitude_weight']*attitude+cfg['omega_weight']*ca.sumsqr(error[10:13]))
            delta = u-(old if block==0 else ca.DM(self.hover)+self.scale*control[:,block-1])
            cost += cfg['input_weight']*ca.sumsqr(control[:,block])+cfg['input_change_weight']*ca.sumsqr(delta/self.scale)
            constraints.append(delta)
        cost += cfg['terminal_factor']*ca.dot(qpos,(s[:3]-ref[:3,-1])**2)
        nlp = {'x':ca.vec(control),'p':ca.vertcat(s0,ca.vec(ref),old),'f':cost,'g':ca.vertcat(*constraints)}
        opts = {'print_time':False,'ipopt.print_level':0,'ipopt.sb':'yes',
                'ipopt.max_iter':cfg['solver_max_iter'],'ipopt.tol':cfg['solver_tolerance'],
                'ipopt.acceptable_tol':cfg['solver_tolerance']*10,'ipopt.acceptable_iter':4,
                'ipopt.hessian_approximation':'limited-memory','ipopt.max_cpu_time':cfg['solver_max_seconds']}
        self.solver = ca.nlpsol('mpc_'+model['name'],'ipopt',nlp,opts)
        self.lbx = np.tile((np.asarray(parameters['command_lower'])-self.hover)/self.scale,self.blocks)
        self.ubx = np.tile((np.asarray(parameters['command_upper'])-self.hover)/self.scale,self.blocks)
        self.warm = np.zeros((4,self.blocks))

    def reset(self): self.warm[:] = 0

    def solve(self,observation,references,previous):
        started = time.perf_counter()
        p = np.r_[observation,references.T.ravel(order='F'),previous]
        try:
            result = self.solver(x0=self.warm.ravel(order='F'),p=p,lbx=self.lbx,ubx=self.ubx,
                                 lbg=-self.cfg['command_delta_per_control'],ubg=self.cfg['command_delta_per_control'])
            stats = self.solver.stats()
            seq = np.asarray(result['x']).ravel().reshape(4,self.blocks,order='F')
            success = bool(stats['success']) and bool(np.isfinite(seq).all())
            status = stats['return_status']
            iterations = int(stats['iter_count'])
            objective = float(result['f']) if np.isfinite(float(result['f'])) else None
        except RuntimeError:
            seq = self.warm.copy(); success=False; status='solver_exception'; iterations=-1; objective=None
        command = self.hover+self.scale*seq[:,0]
        # Accept only a successful, constraint-feasible numerical solution.
        if success:
            commands = self.hover[:,None]+self.scale*seq
            deltas = np.diff(np.column_stack([previous,commands]),axis=1)
            success = bool(np.all(commands>=np.asarray(self.parameters['command_lower'])[:,None]-1e-5)
                           and np.all(commands<=np.asarray(self.parameters['command_upper'])[:,None]+1e-5)
                           and np.max(np.abs(deltas))<=self.cfg['command_delta_per_control']+1e-5)
            if not success: status='constraint_check_failed'
        if success: self.warm = np.column_stack([seq[:,1:],seq[:,-1]])
        return command,{'success':success,'status':status,'seconds':time.perf_counter()-started,
                        'iterations':iterations,'objective':objective}
