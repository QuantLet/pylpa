import warnings

import numpy as np
import scipy

from pylpa.models import ARMAGARCH, GARCH, ARCH


def get_mle_estimator(
        model, data: np.ndarray, maxiter: int = 100, solver: str = "SLSQP",
        weights=None, **kwargs
):
    """

    :param model: 
    :param data:
    :param maxiter:
    :param solver:
    :param weights:
    :param kwargs:
    :return:
    """
    def objective(theta):
        return model.negative_loglikelihood(
            data, theta, weights=weights,
        )

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if solver == "SLSQP" and "bounds" not in kwargs:
        kwargs["bounds"] = model.get_bounds(data)

    result = scipy.optimize.minimize(
        objective, model.initial_params, constraints=model.params_constraints,
        method=solver, options={'maxiter': maxiter, 'disp': False}, **kwargs
    )

    return result


def get_boot_estimator(
        model, data: np.ndarray, weights: np.ndarray, maxtrial: int = 100,
        maxiter: int = 100, **kwargs
):
    """

    :param data:
    :param weights:
    :param maxiter:
    :param model_name:
    :param start_value:
    :param model_kwargs:
    :param kwargs:
    :return:
    """
    success = False
    trial = 0
    while not success:
        trial += 1
        if model.initial_params is None:
            model.initial_params = model.estimate_initial_params(data)
        result = get_mle_estimator(
                model, data, maxiter=maxiter, weights=weights, **kwargs
        )
        if result.message == "Optimization terminated successfully":
            success = True
        if trial > maxtrial:
            break

    return result, success


def get_sup_estimator(A_data, A_weights, B_data, B_weights, A_mle, B_mle,
                      maxtrial, model, **kwargs):
    assert (
            isinstance(model, ARMAGARCH) or isinstance(model, GARCH) or
            isinstance(model, ARCH)
    )
    success = False
    trial = 0
    while not success:
        trial += 1
        model.initial_params = model.estimate_initial_params(
            np.concatenate([A_data, B_data])
        )
        result = sup_estimator(
            model, A_data, A_weights, B_data, B_weights, A_mle, B_mle, **kwargs
        )
        if result.message == "Optimization terminated successfully":
            success = True
        if trial > maxtrial:
            break

    return result, success


def sup_estimator(
        model, A_data, A_weight, B_data, B_weight, A_mle, B_mle, maxiter:
        int = 100, solver: str = "SLSQP",
        **kwargs
):
    def objective(theta):
        bias_theta = theta + B_mle - A_mle
        return (
                model.negative_loglikelihood(
                    A_data, theta, weights=A_weight
                ) + model.negative_loglikelihood(
                B_data, bias_theta, weights=B_weight
            )
        )
    result = scipy.optimize.minimize(
        objective, model.initial_params,
        constraints=model.constraint_sup(A_mle, B_mle),
        options={'maxiter': maxiter, 'disp': False}, method=solver, **kwargs,
    )

    return result
