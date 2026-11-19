"""Optional declared disturbance for compensation validation; never controller input."""
import numpy as np
from pelican_e3.plant import derivative, rotate, step


def disturbance(t, scenario, settings):
    if scenario == 'nominal':
        return np.zeros(6)
    if scenario != 'bias_step':
        raise ValueError(scenario)
    return np.asarray(settings['body_acceleration']) if t >= settings['start_s'] else np.zeros(6)


def disturbed_step(x, u, dt, parameters, integration_dt, bias):
    if not np.any(bias):
        return step(x, u, dt, parameters, integration_dt)
    n = round(dt/integration_dt)
    if not np.isclose(n*integration_dt, dt):
        raise ValueError('Invalid integration grid')
    def rhs(s):
        result = derivative(s, u, parameters)
        result[3:6] += rotate(s[6:10], bias[:3])
        result[10:13] += bias[3:]
        return result
    s = x.copy()
    for _ in range(n):
        a = rhs(s); b = rhs(s+.5*integration_dt*a)
        c = rhs(s+.5*integration_dt*b); d = rhs(s+integration_dt*c)
        s += integration_dt*(a+2*b+2*c+d)/6
        s[6:10] /= np.linalg.norm(s[6:10])
    return s
