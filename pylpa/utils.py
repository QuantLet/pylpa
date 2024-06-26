import numpy as np

from pylpa.models.garchtools import garch_process


def get_max_k(multiplier, n_0, T):
    max_k = int(np.log(T / n_0) / np.log(multiplier))
    return max_k


def generate_garch_data(
        omega1=0.2, a1=0.2, b1=0.1, n1=100, omega2=1, a2=0.2, b2=0.7, n2=50
):
    # Simulating a GARCH(1, 1) process
    Y1, sigsq1 = garch_process(n1, omega1, a1, b1)
    Y2, sigsq2 = garch_process(n2, omega2, a2, b2)
    Y = np.concatenate([Y1, Y2, Y1, Y2, Y1, Y2])
    sigsq = np.concatenate([sigsq1, sigsq2, sigsq1, sigsq2, sigsq1, sigsq2])
    breakpoints = [n1, n1 + n2, 2 * n1 + n2, 2 * (n1 + n2), 3 * n1 + 2 * n2,
                   3 * (n1 + n2)]
    return Y, sigsq, breakpoints


def default_config(config: dict):
    if 'save_k' not in config.keys():
        config['save_k'] = False
    if 'solver' not in config.keys():
        config['solver'] = 'SLSQP'
    if 'maxiter' not in config.keys():
        config['maxiter'] = 100
    if 'generate' not in config.keys():
        config['generate'] = 'normal'

    return config
