"""Independent continuous Newton-Euler model with effective calibrated gains.

No sparse-regression coefficients or SINDy features enter the simulator.
Rotor code is a latent actuator coordinate; it is not physical rad/s.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def rotate(q,v,inverse=False):
    xyz = -q[..., :3] if inverse else q[..., :3]
    a = 2*np.cross(xyz,v)
    return v+q[...,3:4]*a+np.cross(xyz,a)


def derivative(state,u,parameters):
    q = state[...,6:10]
    w = state[...,10:13]
    m = state[...,13:17]
    body_v = rotate(q,state[...,3:6],True)
    force = -np.asarray(parameters['linear_drag'])*body_v
    force[...,2] += (m/100)**2@np.asarray(parameters['thrust_coefficients'])
    acc = rotate(q,force)
    acc[...,2] -= parameters['gravity']
    inertia = np.asarray(parameters['inertia_ratio'])
    alpha = (m/100)**2@np.asarray(parameters['torque_matrix']).T
    alpha += body_v@np.asarray(parameters.get('aero_moment_matrix',np.zeros((3,3)))).T
    alpha -= np.asarray(parameters['angular_drag'])*w+np.cross(w,inertia*w)/inertia
    qdot = .5*np.concatenate([q[...,3:4]*w+np.cross(q[...,:3],w),
                              -np.sum(q[...,:3]*w,axis=-1,keepdims=True)],axis=-1)
    steady = np.asarray(parameters['motor_gain'])*u+np.asarray(parameters['motor_bias'])
    mdot = (steady-m)/np.asarray(parameters['motor_tau'])
    return np.concatenate([state[...,3:6],acc,qdot,alpha,mdot],axis=-1)


def step(state,u,dt,parameters,integration_dt=.0025):
    count = round(dt/integration_dt)
    if not np.isclose(count*integration_dt,dt): raise ValueError('Invalid integration grid')
    x = np.asarray(state).copy()
    for _ in range(count):
        k1 = derivative(x,u,parameters)
        k2 = derivative(x+.5*integration_dt*k1,u,parameters)
        k3 = derivative(x+.5*integration_dt*k2,u,parameters)
        k4 = derivative(x+integration_dt*k3,u,parameters)
        x += integration_dt*(k1+2*k2+2*k3+k4)/6
        x[...,6:10] /= np.linalg.norm(x[...,6:10],axis=-1,keepdims=True)
    return x


def observe(previous,current,dt):
    result = current[:13].copy()
    result[3:6] = (current[:3]-previous[:3])/dt
    result[10:13] = (Rotation.from_quat(previous[6:10]).inv()*Rotation.from_quat(current[6:10])).as_rotvec()/dt
    return result


def allocation(parameters):
    return np.vstack([parameters['thrust_coefficients'],parameters['torque_matrix']])


def steady_command(force_and_acceleration,parameters):
    squared = np.linalg.solve(allocation(parameters),force_and_acceleration)
    # PD requested wrench can be infeasible. Final common command bounds are
    # applied by the runner and counted. Negative requested rotor force -> zero.
    motor = 100*np.sqrt(np.maximum(squared,0))
    return (motor-np.asarray(parameters['motor_bias']))/np.asarray(parameters['motor_gain'])
