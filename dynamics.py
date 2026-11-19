"""Body-frame acceleration targets, pose-derived rates and discrete kinematics."""
import numpy as np
from scipy.spatial.transform import Rotation


def rotate(q, v, inverse=False):
    xyz = -q[..., :3] if inverse else q[..., :3]
    t = 2*np.cross(xyz, v)
    return v+q[..., 3:4]*t+np.cross(xyz, t)


def qmultiply(a, b):
    xyz = a[..., 3:4]*b[..., :3]+b[..., 3:4]*a[..., :3]+np.cross(a[..., :3], b[..., :3])
    scalar = a[..., 3]*b[..., 3]-np.sum(a[..., :3]*b[..., :3], axis=-1)
    return np.concatenate([xyz, scalar[..., None]], axis=-1)


def qexp(rotvec):
    angle = np.linalg.norm(rotvec, axis=-1)
    factor = .5*np.sinc(angle/(2*np.pi))
    return np.concatenate([rotvec*factor[..., None], np.cos(angle/2)[..., None]], axis=-1)


def prepare_flight(raw, flight_id, dt):
    position = np.asarray(raw['Pos'], dtype=float)
    orientation = Rotation.from_euler('xyz', raw['Euler'])
    quaternion = orientation.as_quat()
    sign = np.r_[1, np.cumprod(np.where(np.sum(quaternion[1:]*quaternion[:-1], axis=1) < 0, -1, 1))]
    quaternion *= sign[:, None]
    velocity = np.diff(position, axis=0)/dt
    omega = (orientation[:-1].inv()*orientation[1:]).as_rotvec()/dt
    x = np.c_[position[1:], velocity, quaternion[1:], omega]
    u = np.asarray(raw['Motors_CMD'], dtype=float)[1:].copy()
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError('Nonfinite prepared state; no silent imputation')
    return {'id': int(flight_id), 'x': x, 'u': u,
            't': np.arange(1, len(position))*dt,
            'raw_index': np.arange(1, len(position), dtype=np.int64)}


def state_features(x, gravity):
    vbody = rotate(x[..., 6:10], x[..., 3:6], inverse=True)
    gvec = np.zeros_like(vbody)
    gvec[..., 2] = -gravity
    gbody = rotate(x[..., 6:10], gvec, inverse=True)
    return np.concatenate([vbody, x[..., 10:13], gbody], axis=-1)


def target_acceleration(episode, indices, config):
    x = episode['x']
    acc = (x[indices+1, 3:6]-x[indices, 3:6])/config['dt']
    acc[:, 2] += config['gravity']
    force_body = rotate(x[indices, 6:10], acc, inverse=True)
    angular = (x[indices+1, 10:13]-x[indices, 10:13])/config['dt']
    return np.c_[force_body, angular]


def advance(x, acceleration, config):
    """Kinematics consistent with end-stamped backward-difference state labels."""
    a_world = rotate(x[..., 6:10], acceleration[..., :3])
    a_world[..., 2] -= config['gravity']
    result = x.copy()
    result[..., 3:6] = x[..., 3:6]+config['dt']*a_world
    result[..., 10:13] = x[..., 10:13]+config['dt']*acceleration[..., 3:]
    result[..., :3] = x[..., :3]+config['dt']*result[..., 3:6]
    result[..., 6:10] = qmultiply(x[..., 6:10], qexp(config['dt']*result[..., 10:13]))
    result[..., 6:10] /= np.linalg.norm(result[..., 6:10], axis=-1, keepdims=True)
    return result
