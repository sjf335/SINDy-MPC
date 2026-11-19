"""Smooth-start circle, figure-eight and ascending helix references."""
import numpy as np
from scipy.spatial.transform import Rotation


def translation(t,kind,cfg):
    t = max(0,float(t))
    ramp,w,r = cfg['trajectory_ramp_s'],cfg['trajectory_frequency_rad_s'],cfg['trajectory_radius_m']
    if t<ramp:
        s = t/ramp
        phase = w*ramp*(s**3-.5*s**4)
        rate = w*(3*s*s-2*s**3)
        acceleration = w*(6*s-6*s*s)/ramp
    else:
        phase,rate,acceleration = w*(t-.5*ramp),w,0.
    if kind in ['circle','helix']:
        p = np.array([r*(np.cos(phase)-1),r*np.sin(phase),1.])
        d = np.array([-r*np.sin(phase),r*np.cos(phase),0.])
        dd = np.array([-r*np.cos(phase),-r*np.sin(phase),0.])
    elif kind=='figure8':
        p = np.array([r*np.sin(phase),.5*r*np.sin(2*phase),1.])
        d = np.array([r*np.cos(phase),r*np.cos(2*phase),0.])
        dd = np.array([-r*np.sin(phase),-2*r*np.sin(2*phase),0.])
    elif kind=='hover':
        return np.array([0.,0.,1.]),np.zeros(3),np.zeros(3)
    else: raise ValueError(kind)
    if kind=='helix':
        climb = cfg['helix_vertical_speed_m_s']/w
        p[2] += climb*phase
        d[2] = climb
    return p,d*rate,dd*rate*rate+d*acceleration


def rotation_from_force(force,yaw=0.):
    b3 = force/np.linalg.norm(force)
    heading = np.array([np.cos(yaw),np.sin(yaw),0.])
    b2 = np.cross(b3,heading)
    b2 /= np.linalg.norm(b2)
    b1 = np.cross(b2,b3)
    return Rotation.from_matrix(np.column_stack([b1,b2,b3]))


def at(t,kind,cfg):
    p,v,a = translation(t,kind,cfg)
    r = rotation_from_force(a+np.array([0,0,cfg['gravity']]))
    eps = 1e-4
    ap = translation(t+eps,kind,cfg)[2]
    rp = rotation_from_force(ap+np.array([0,0,cfg['gravity']]))
    omega = (r.inv()*rp).as_rotvec()/eps
    return np.r_[p,v,r.as_quat(),omega],a
