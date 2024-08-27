import time
from typing import List, Optional

from pylpa.constant import MULTIPLIER, N_0, MAX_INTERVAL_LENGTH
from pylpa.logger import LOGGER

import numpy as np

from pylpa.estimators import (
    get_boot_estimator,
    get_sup_estimator,
    get_mle_estimator
)
import pickle
from joblib import Parallel, delayed

from pylpa.models.utils import build_model_from_config
from pylpa.utils import get_max_k


def bootstrap_test(
        i, model, A_k, B_k, MLE_A, MLE_B, maxtrial, solver: str = "SLSQP",
        bounds=None, maxiter=100, generate='normal', **kwargs,
):
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

    MLE_A_b, A_success = get_boot_estimator(
        model, A_k, weights_A, maxtrial, **kwargs
    )
    LL_MLE_A_b = - 1.0 * MLE_A_b.fun  # -1 since we minimized negative log likelihood

    if not A_success:
        LL_MLE_A_b = np.nan
        LOGGER.warning(
            'Estimation failed on A for simulation nb %s, message:' % str(i))
        LOGGER.warning(MLE_A_b.message)
        LOGGER.warning('LL value: %s' % str(- np.round(MLE_A_b.fun, 2)))

    MLE_B_b, B_success = get_boot_estimator(
        model, B_k, weights_B, maxtrial, **kwargs
    )
    LL_MLE_B_b = - 1.0 * MLE_B_b.fun  # -1 since we minimized negative log likelihood
    if not B_success:
        LL_MLE_B_b = np.nan
        LOGGER.warning(
            'Estimation failed on B for simulation nb %s, message:' % str(i))
        LOGGER.warning(MLE_B_b.message)
        LOGGER.warning('LL value: %s' % str(- np.round(MLE_B_b.fun, 2)))

    sup_result, sup_success = get_sup_estimator(
        A_k, weights_A, B_k, weights_B, MLE_A.x, MLE_B.x, maxtrial, model,
        solver=solver, maxiter=maxiter, **kwargs
    )

    LL_sup_result = - 1.0 * sup_result.fun  # -1 since we minimized negative log likelihood
    if not sup_success:
        LL_sup_result = np.nan
        LOGGER.warning(
            'Estimation failed on sup for simulation nb %s, message:' % str(i))
        LOGGER.warning(sup_result.message)
        LOGGER.warning('LL value: %s' % str(- np.round(sup_result.fun, 2)))

    T_k_b = LL_MLE_A_b + LL_MLE_B_b - LL_sup_result

    return T_k_b


def test_interval(
        model, k: int, data: np.ndarray, n_ks: Optional[List] = None,
        T=None, num_sim: int = 100, min_steps: int = 1,
        level: float = 0.95, maxtrial: int = 10,
        save_dir: Optional[str] = None, generate: str = 'normal',
        njobs: int = 8, solver: str = 'SLSQP', maxiter: int = 100, **kwargs
):
    """
    Test whether the interval at n_ks[k] in the data contains a breakpoint.
    If min_steps=1 all points are tested
    :param k: interval index
    :param data: data
    :param model_name
    :param n_ks: length of intervals to test
    :param num_sim: bootstrap simulation
    :param min_steps: min distance between two break point test
    :param level: critical value level
    :param maxtrial: maximum attempts to estimate MLE
    :return: dict
    """

    assert T is not None
    assert T <= len(data)

    bounds = model.get_bounds(data)

    result_test = {}
    t0_k = time.time()
    # started from 1 because 0th interval already homogeneous and K+1
    # because last point is not inclusive
    LOGGER.info('Interval nb %s' % k)
    LOGGER.info('T is: %s' % T)

    n_k_minus1 = n_ks[0]
    n_k = n_ks[1]
    n_k_plus1 = n_ks[2]
    assert n_k_plus1 > n_k

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

    # if preprocessing is not None:
    #     if preprocessing.get("name") == "StandardScaler":
    #         I_k = (I_k - np.mean(I_k)) / np.std(I_k)
    #         I_k_plus1 = (I_k_plus1 - np.mean(I_k_plus1)) / np.std(I_k_plus1)

    model.initial_params = model.estimate_initial_params(I_k_plus1)
    MLE_I_k_plus1 = get_mle_estimator(
        model, I_k_plus1, maxiter=maxiter, solver=solver, bounds=bounds,
    )

    assert start_k < T - n_k_minus1
    J_k = np.array(range(start_k, T - n_k_minus1, min_steps))[::-1].tolist()
    T_k = np.zeros(len(J_k))
    T_k_b = np.empty((len(J_k), num_sim))

    t0b = time.time()
    for counter, s in enumerate(J_k):
        LOGGER.info('Break points to go: %s' % str(len(J_k) - counter))
        # New intervals
        A_k = data[start_k_plus1:(s + 1)]
        B_k = data[s + 1:T]

        # Estimator
        model.initial_params = model.estimate_initial_params(I_k)
        MLE_A = get_mle_estimator(
            model, A_k, maxiter=maxiter, solver=solver,
            bounds=bounds, **kwargs
        )
        MLE_B = get_mle_estimator(
            model, B_k, maxiter=maxiter, solver=solver,
            bounds=bounds, **kwargs
        )

        # test statistic
        T_k[counter] = -1.0 * (
                MLE_A.fun + MLE_B.fun - MLE_I_k_plus1.fun)

        # Run bootstrapt tests
        def runner(i):
            return bootstrap_test(
                i, model, A_k, B_k, MLE_A, MLE_B, maxtrial, solver=solver,
                bounds=bounds, maxiter=maxiter, generate=generate
            )

        if njobs == 1:
            results = []
            for i in range(num_sim):
                r = runner(i)
                results.append(r)
        else:
            results = Parallel(
                n_jobs=njobs
            )(delayed(runner)(i) for i in range(num_sim))
        T_k_b[counter, :] = results

    t1b = time.time()

    LOGGER.info('Time for jk: %s' % str(round((t1b - t0b) / 60, 2)))

    boot_test = np.nanmax(T_k_b, axis=0)

    LOGGER.info('boot_test: %s' % boot_test)
    LOGGER.info('T_k: %s' % T_k)
    assert len(boot_test.shape) == 1
    assert boot_test.shape[0] == num_sim

    test_value = max(T_k)
    critical_value = np.sqrt(2.0 * np.quantile(boot_test, level))
    null_is_true = test_value <= critical_value
    if np.isnan(critical_value):
        LOGGER.warning("Critical value is NaN!!")
    else:
        LOGGER.info('test_value: %s' % test_value)
        LOGGER.info('critical_value: %s' % critical_value)
        LOGGER.info('NULL REJECTED: %s' % str(not null_is_true))

    if not null_is_true:
        index = J_k[np.argmax(T_k)]
        LOGGER.info('Break point detected at index: %s' % str(index))
        I_window = data[index:T]

        model.initial_params = model.estimate_initial_params(I_window)
        MLE_window = get_mle_estimator(
            model, I_window, maxiter, solver=solver,
            bounds=bounds, **kwargs
        )
        window = len(I_window)
        scaled_window = len(I_window) / T
    else:
        model.initial_params = model.estimate_initial_params(I_k)
        MLE_window = get_mle_estimator(
            model, I_k, maxiter, solver=solver,
            bounds=bounds, **kwargs
        )
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
        pickle.dump(result_test, open(
            '%s/lpa_result_%s_T_%s.p' % (save_dir, str(k), str(T)), 'wb'))

    t1_k = time.time()
    LOGGER.info('TIME IN MIN FOR INTERVAL %s: %s' % (
    k, str(np.round((t1_k - t0_k) / 60, 2))))

    return result_test, null_is_true, last_test


def generate_interval_indices(
        N, interval_step: int = None, K: Optional[int] = None):
    if interval_step is None:
        if K is None:
            K = get_max_k(MULTIPLIER, N_0, N)
        n_ks = [[int(np.round(N_0 * MULTIPLIER ** (k - 1))),
                 int(np.round(N_0 * MULTIPLIER ** k)),
                 int(np.round(N_0 * MULTIPLIER ** (k + 1)))] for k in
                range(1, K+1)]
    else:
        if N + interval_step > MAX_INTERVAL_LENGTH:
            max_int = MAX_INTERVAL_LENGTH + interval_step
        else:
            max_int = N
        n_ks = list(range(N_0, max_int, interval_step))
        n_ks = [[n_ks[i - 1], n_ks[i], n_ks[i + 1]] for i in
                range(1, len(n_ks) - 1)]
    return n_ks


def find_largest_homogene_interval(
        data: np.ndarray, model_config: dict,  K: Optional[int] = None,
        interval_step: Optional[int] = None, num_sim: int = 100,
        min_steps: int = 1, maxtrial: int = 10, generate: str = 'normal',
        solver: str = 'SLSQP', maxiter: int = 100,
        njobs: int = 8, **kwargs):
    """

    :param data:
    :param K:
    :param interval_step: Frequency of test
    :param kwargs:
    :return:
    """
    # Algo
    n_ks = generate_interval_indices(
        len(data), interval_step=interval_step,  K=K
    )
    LOGGER.info(f"Create model {model_config['name']}")
    model = build_model_from_config(model_config)

    LOGGER.info('Find largest window')
    LOGGER.info('Candidate windows: %s' % n_ks)
    for k in range(len(n_ks)):
        res_k, null_is_true, last_test = test_interval(
            model, k, data, n_ks=n_ks[k],
            T=len(data), num_sim=num_sim,
            min_steps=min_steps, maxtrial=maxtrial, njobs=njobs,
            solver=solver, maxiter=maxiter, generate=generate, **kwargs,
        )
        if not null_is_true:
            index = res_k['J_k'][np.argmax(res_k['T_k'])]
            LOGGER.info('Break point detected at index: %s' % str(index))
            LOGGER.info(f"n_ks: {n_ks[k]}")
            return data[-n_ks[k][1]:], index
        else:
            index = np.min(res_k['J_k'])
            assert index == res_k['J_k'][-1]
            return data, -1