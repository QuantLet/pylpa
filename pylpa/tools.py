import numpy as np
from pylpa.garch.garch import (
    starting_values,
    get_bounds,
    mle_estimator,
    get_start_value,
    sup_estimator,
)


def residual_bootstrap(yhat, residuals, num_sim=10):
    assert len(yhat.shape) == 1
    epsFixed = np.concatenate([residuals, - residuals])

    def bootIter():
        epsBootstrap = np.random.choice(epsFixed, yhat.shape[0])
        boot_sample = yhat + epsBootstrap
        return boot_sample

    boot_sample = np.concatenate(
        [np.array(bootIter()).reshape(-1, 1) for _ in range(num_sim)], axis=1)

    return boot_sample


def get_starting_values(data, model_name, **kwargs):
    if model_name in ['garch', 'arch', 'arma-garch', 'arma-arch']:
        start_value = starting_values(data, **kwargs)
    elif model_name == 'carlInd':
        start_value = [0.01, 0.2, 0.03]
    elif model_name == 'carlVol':
        start_value = starting_values(data, garch=True)
        start_value = [0.01, 0.2] + start_value[1:]

    return start_value


def get_mle_estimator(data, start_value, model_name, maxiter, solver=None,
                      **kwargs):
    if model_name in ['garch', 'arch']:
        arma = kwargs['arma']
        if 'bounds' not in kwargs:
            if solver == 'SLSQP':
                if model_name == 'garch':
                    bounds = get_bounds(data, arma=arma)
                else:
                    bounds = get_bounds(data, q=0, arma=arma)
            else:
                bounds = None
        else:
            bounds = kwargs['bounds']
        MLE = mle_estimator(
            data, start_value, arma=arma, method=solver, bounds=bounds,
            options={'maxiter': maxiter, 'disp': False}
        )
    else:
        raise NotImplementedError
    return MLE


def get_boot_estimator(
        data: np.ndarray, weights, max_iter, model_name, start_value=None,
        model_kwargs=None, **kwargs
):
    """

    :param data:
    :param weights:
    :param max_iter:
    :param model_name:
    :param start_value:
    :param model_kwargs:
    :param kwargs:
    :return:
    """
    if model_name in ['garch', 'arch', 'arma-garch', 'arma-arch']:
        garch = model_kwargs['garch']
        arma = model_kwargs['arma']
        success = False
        iter_ = 0
        while not success:
            iter_ += 1
            if start_value is None:
                start_value = get_starting_values(
                    data, model_name, arch=garch, arma=arma
                )
            result = mle_estimator(
                data, start_value, arma=arma, weights=weights, **kwargs
            )
            if result.message == "Optimization terminated successfully":
                success = True
            if iter_ > max_iter:
                break
    else:
        raise NotImplementedError

    return result, success


def get_sup_estimator(A_data, A_weights, B_data, B_weights, A_mle, B_mle,
                      max_iter, model_name, start_value=None,
                      model_kwargs=None, **kwargs):
    if model_name in ['garch', 'arch', 'arma-garch', 'arma-arch']:
        garch = model_kwargs['garch']
        arma = model_kwargs['arma']
        success = False
        iter_ = 0
        while not success:
            iter_ += 1
            if start_value is None:
                start_value = starting_values(
                    np.concatenate([A_data, B_data]), garch=garch, arma=arma
                )
            start_value = get_start_value(start_value, A_mle, B_mle, arma)
            result = sup_estimator(
                A_data, A_weights, B_data, B_weights, start_value, A_mle,
                B_mle, arma=arma, **kwargs
            )
            if result.message == "Optimization terminated successfully":
                success = True
            if iter_ > max_iter:
                break
    else:
        raise ValueError()

    return result, success
