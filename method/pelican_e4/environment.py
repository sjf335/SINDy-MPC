"""Physical interventions and causal pose observations, never controller inputs."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from pelican_e3.plant import derivative, step, observe


def plant_parameters(nominal, scenario):
    result = deepcopy(nominal)
    result['motor_tau'] = (np.asarray(nominal['motor_tau'])*scenario['tau_scale']).tolist()
    return result


def forcing(t, scenario, settings):
    if settings['force_start_s'] <= t < settings['force_end_s']:
        return np.asarray(scenario['world_acceleration_m_s2'], dtype=float)
    return np.zeros(3)


def physical_step(x, u, dt, parameters, integration_dt, acceleration):
    if not np.any(acceleration):
        return step(x, u, dt, parameters, integration_dt)
    count = round(dt/integration_dt)
    if not np.isclose(count*integration_dt, dt):
        raise ValueError('Invalid integration grid')
    def rhs(s):
        dx = derivative(s, u, parameters)
        dx[3:6] += acceleration
        return dx
    state = x.copy()
    for _ in range(count):
        a = rhs(state); b = rhs(state+.5*integration_dt*a)
        c = rhs(state+.5*integration_dt*b); d = rhs(state+integration_dt*c)
        state += integration_dt*(a+2*b+2*c+d)/6
        state[6:10] /= np.linalg.norm(state[6:10])
    return state


def noise_samples(seed, count, scenario, settings):
    rng = np.random.default_rng(settings['noise_seed_offset']+int(seed))
    # Generate a fixed standardized trace for pairing across controller/trajectory.
    return rng.standard_normal((count, 6))*np.r_[np.full(3, scenario['position_noise_std_m']),
                                               np.full(3, scenario['rotation_noise_std_rad'])]


def measured_pose(truth, noise):
    result = np.asarray(truth[:13]).copy()
    if not np.any(noise):
        return result
    result[:3] += noise[:3]
    result[6:10] = (Rotation.from_quat(truth[6:10])*Rotation.from_rotvec(noise[3:])).as_quat()
    return result


def observation_history(initial, noise, dt):
    # Derivatives in the first row are not consumed: all retained lags lie inside warmup.
    states = initial['states'].copy()
    poses = [measured_pose(x, e) for x, e in zip(initial['truth'], noise)]
    states[0, :3] = poses[0][:3]; states[0, 6:10] = poses[0][6:10]
    for k in range(1, len(states)):
        states[k] = observe(poses[k-1], poses[k], dt)
    return states, poses[-1]


def event_metrics(t, errors, complete, settings):
    force = (t >= settings['force_start_s']) & (t <= settings['force_end_s'])
    after = t >= settings['post_event_start_s']
    result = dict(event_peak_error_m=float(errors[force].max()) if force.any() else None,
                  post_event_rmse_m=float(np.sqrt(np.mean(errors[after]**2))) if complete and after.any() else None,
                  recovered=False, recovery_time_s=None)
    dt = t[1]-t[0] if len(t) > 1 else 0.01
    window = round(settings['recovery_dwell_s']/dt)+1
    for k in np.flatnonzero(t >= settings['force_end_s']):
        if k+window > len(t): break
        if np.all(errors[k:k+window] <= settings['recovery_error_m']):
            result.update(recovered=True, recovery_time_s=float(t[k]-settings['force_end_s']))
            break
    return result
