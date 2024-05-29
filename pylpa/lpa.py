import time

from pylpa.logger import LOGGER

import numpy as np

from pylpa.tools import (
    get_boot_estimator,
    get_starting_values,
    get_sup_estimator, get_mle_estimator
)
import pickle
from joblib import Parallel, delayed
from pylpa.garch.garch import get_bounds


# Calculating estimators
def one_bootstrap_test(i, A_k, B_k, MLE_A, MLE_B, max_trial, model_name, method, bounds, options,
                       start_value=None, model_kwargs=None, generate='normal'):

    kwargs = {'method': method, 'options': options}
    if bounds is not None:
        kwargs['bounds'] = bounds

    # generating weights
    if generate == 'normal':
        weights_A = np.random.normal(loc=1.0, scale=1.0, size=len(A_k))
        weights_B = np.random.normal(loc=1.0, scale=1.0, size=len(B_k))
    elif generate == 'exponential':
        weights_A = np.random.exponential(1, size=len(A_k))
        weights_B = np.random.exponential(1, size=len(B_k))
    else:
        raise ValueError

    MLE_A_b, A_success = get_boot_estimator(A_k, weights_A, max_trial, model_name, start_value=start_value,
                                            model_kwargs=model_kwargs, **kwargs) # method=method, bounds=bounds, options=options)
    LL_MLE_A_b = - 1.0 * MLE_A_b.fun  # -1 since we minimized negative log likelihood

    if not A_success:
        LL_MLE_A_b = np.nan
        LOGGER.info('Estimation failed on A for simulation nb %s, message:' % str(i))
        LOGGER.warning(MLE_A_b.message)
        LOGGER.warning('LL value: %s' % str(- np.round(MLE_A_b.fun, 2)))

    MLE_B_b, B_success = get_boot_estimator(B_k, weights_B, max_trial, model_name, start_value=start_value,
                                            model_kwargs=model_kwargs, **kwargs)
    LL_MLE_B_b = - 1.0 * MLE_B_b.fun  # -1 since we minimized negative log likelihood
    if not B_success:
        LL_MLE_B_b = np.nan
        LOGGER.warning('Estimation failed on B for simulation nb %s, message:' % str(i))
        LOGGER.warning(MLE_B_b.message)
        LOGGER.warning('LL value: %s' % str(- np.round(MLE_B_b.fun, 2)))

    sup_result, sup_success = get_sup_estimator(A_k, weights_A, B_k, weights_B, MLE_A.x, MLE_B.x, max_trial, model_name,
                                                start_value=start_value, model_kwargs=model_kwargs, **kwargs)

    LL_sup_result = - 1.0 * sup_result.fun  # -1 since we minimized negative log likelihood
    if not sup_success:
        LL_sup_result = np.nan
        LOGGER.warning('Estimation failed on sup for simulation nb %s, message:' % str(i))
        LOGGER.warning(sup_result.message)
        LOGGER.warning('LL value: %s' % str(- np.round(sup_result.fun, 2)))

    T_k_b = LL_MLE_A_b + LL_MLE_B_b - LL_sup_result

    return T_k_b


def test_one_interval(k, data, model_name, n_ks=None, T=None, num_sim=2, min_steps=4, multiplier=1.20, n_0=60,
                      level=0.95, max_trial=10,
                      save_dir=None, generate='normal', njobs=80, solver='SLSQP', maxiter=100, mean_std_norm=False, **kwargs):
    """
    :param k: interval index
    :param data: data
    :param num_sim: bootstrap simulation
    :param min_steps: min distance between two break point test
    :param multiplier: interval length multiplier
    :param n_0: minimum homogeneity length
    :param level: critical value level
    :param max_trial: maximum attempts to estimate MLE
    :return: dict
    """

    assert T is not None
    assert T <= len(data)

    if model_name in ["arch", "garch"]:
        if solver == 'SLSQP':
            garch = kwargs['garch']
            arma = kwargs['arma']
            if garch:
                bounds = get_bounds((data-np.mean(data))/np.std(data), arma=arma) if mean_std_norm else get_bounds(data, arma=arma)
            else:
                bounds = get_bounds((data-np.mean(data))/np.std(data), q=0, arma=arma) if mean_std_norm else get_bounds(data, q=0, arma=arma)
        if garch:
            p = 1
            q = 1
        else:
            p = 1
            q = 0
    else:
        bounds = None

    result_test = {}
    t0_k = time.time()
    # started from 1 because 0th interval already homogeneous and K+1 because last point is not inclusive
    LOGGER.info('Interval nb %s' % k)
    LOGGER.info('T is: %s' % T)

    n_k_minus1 = n_ks[0]
    n_k = n_ks[1]
    n_k_plus1 = n_ks[2]

    """
    n_k = np.int(np.round(n_0 * multiplier ** k))
    n_k_plus1 = np.int(n_0 * multiplier ** (k + 1))
    """
    assert n_k_plus1 > n_k

    """
    if k > 0:
        n_k_minus1 = np.int(np.round(n_0 * multiplier ** (k - 1)))
    else:
        n_k_minus1 = n_0
    """

    if T - n_k >= 0:
        start_k = T - n_k
    else:
        start_k = 0

    if T - n_k_plus1 >= 0:
        start_k_plus1 = T - n_k_plus1
        last_test = False
    else:
        start_k_plus1 = 0
        last_test = True

    assert start_k_plus1 < start_k

    I_k = data[start_k:T]
    I_k_plus1 = data[start_k_plus1:T]

    if mean_std_norm:
        I_k = (I_k-np.mean(I_k)) / np.std(I_k)
        I_k_plus1 = (I_k_plus1-np.mean(I_k_plus1)) / np.std(I_k_plus1)


    #print(I_k)
    #print(I_k.shape)
    #print(I_k_plus1.shape)
    #exit()
    # start_value = starting_values(I_k_plus1, p=p, q=q)

    # start_value = starting_values(I_k_plus1, garch=garch, arma=arma)
    start_value = get_starting_values(I_k_plus1, model_name, **kwargs)
    MLE_I_k_plus1 = get_mle_estimator(I_k_plus1, start_value, model_name, maxiter, solver=solver, bounds=bounds,
                                      **kwargs)

    assert start_k < T - n_k_minus1
    J_k = range(start_k, T - n_k_minus1)
    J_k = np.array(list(J_k))[list(range(0, len(J_k), min_steps))]

    T_k = np.zeros(len(J_k))
    T_k_b = np.empty((len(J_k), num_sim))

    t0b = time.time()
    for counter, s in enumerate(J_k):
        LOGGER.info('Break points to go: %s' % str(len(J_k) - counter))

        # New intervals
        A_k = data[start_k_plus1:(s + 1)]
        B_k = data[s + 1:T]

        # Estimator
        start_value = get_starting_values(I_k, model_name, **kwargs)
        MLE_A = get_mle_estimator(A_k, start_value, model_name, maxiter, solver=solver, bounds=bounds, **kwargs)

        start_value = get_starting_values(I_k, model_name, **kwargs)
        MLE_B = get_mle_estimator(B_k, start_value, model_name, maxiter, solver=solver, bounds=bounds, **kwargs)

        # test statistic
        T_k[counter] = -1.0 * (
                MLE_A.fun + MLE_B.fun - MLE_I_k_plus1.fun)  # -1 since we minimized negative log likelihood

        # Run bootstrapt tests
        start_value = get_starting_values(np.concatenate([A_k, B_k]), model_name, **kwargs)

        def runner(i):
            return one_bootstrap_test(i, A_k, B_k, MLE_A, MLE_B, max_trial, model_name, solver, bounds, {'maxiter': maxiter},
                                      start_value=start_value, model_kwargs=kwargs, generate=generate)
        results = Parallel(n_jobs=njobs)(delayed(runner)(i) for i in range(num_sim))

        """
        results = []
        for i in range(num_sim):
            results.append(runner(i))
        """
        T_k_b[counter, :] = results

    t1b = time.time()

    LOGGER.info('Time for jk: %s' % str(round((t1b - t0b) / 60, 2)))

    boot_test = np.nanmax(T_k_b, axis=0)
    LOGGER.info('####  boot_test: %s' % boot_test)
    LOGGER.info('####  T_k: %s' % T_k)
    assert len(boot_test.shape) == 1
    assert boot_test.shape[0] == num_sim

    test_value = max(T_k)
    critical_value = np.sqrt(2.0 * np.quantile(boot_test, level))
    null_is_true = test_value <= critical_value
    LOGGER.info('####  test_value: %s' % test_value)
    LOGGER.info('####  critical_value: %s' % critical_value)
    LOGGER.info('#### NULL REJECTED: %s' % str(not null_is_true))

    if not null_is_true:
        index = J_k[np.argmax(T_k)]
        LOGGER.info('##### break point detected at index: %s' % str(index))
        I_window = data[index:T]
        start_value = get_starting_values(I_window, model_name, **kwargs)

        MLE_window = get_mle_estimator(I_window, start_value, model_name, maxiter, solver=solver, bounds=bounds, **kwargs)
        window = len(I_window)
        scaled_window = len(I_window) / T
    else:
        start_value = get_starting_values(I_k, model_name, **kwargs)
        MLE_window = get_mle_estimator(I_k, start_value, model_name, maxiter, solver=solver, bounds=bounds, **kwargs)
        window = len(I_k)
        scaled_window = len(I_k) / T

    result_test['T_k'] = T_k
    result_test['J_k'] = J_k
    result_test['boot_test'] = boot_test
    result_test['test_value'] = test_value
    result_test['critical_value'] = critical_value
    result_test['H0'] = null_is_true
    result_test['MLE'] = MLE_window
    result_test['window'] = window
    result_test['end_k_minus1'] = T - n_k_minus1
    result_test['scaled_window'] = scaled_window

    if save_dir is not None:
        pickle.dump(result_test, open('%s/lpa_result_%s_T_%s.p' % (save_dir, str(k), str(T)), 'wb'))

    t1_k = time.time()
    LOGGER.info('##### TIME IN MIN FOR INTERVAL %s: %s' % (k, str(np.round((t1_k - t0_k) / 60, 2))))

    return result_test, null_is_true, last_test


