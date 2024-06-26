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


def default_bootstrat_config(config: dict):
    if "bootstrap" not in config:
        bootstrap = {
            "generate": "normal",
            "num_sim": 100,
            "njobs": 8
        }
    else:
        bootstrap = config["bootstrap"]
        if "generate" not in bootstrap:
            bootstrap["generate"] = "normal"
        if "num_sim" not in bootstrap:
            bootstrap["num_sim"] = 100
        if "njobs" not in bootstrap:
            bootstrap["njobs"] = 8

    return bootstrap


def default_config(config: dict):
    if 'solver' not in config.keys():
        config['solver'] = 'SLSQP'
    if 'maxiter' not in config.keys():
        config['maxiter'] = 100
    if "K" not in config:
        config["K"] = None
    if "interval_step" not in config:
        config["interval_step"] = None
    config["bootstrap"] = default_bootstrat_config(config)

    return config
