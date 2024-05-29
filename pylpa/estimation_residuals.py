import logging
import time
from surrogate.tools import *
import pickle
from joblib import Parallel, delayed


def test_one_interval(k, data, T=None, num_sim=2, min_steps=4, multiplier=1.20, n_0=60, level=0.95, max_trial=10, save_dir=None,
                      garch=False, generate='normal', njobs=80, solver='SLSQP', maxiter=100):
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

    if solver == 'SLSQP':
        if garch:
            bounds = get_bounds(data)
        else:
            bounds = get_bounds(data, q=0)
    else:
        bounds = None
    if garch:
        p = 1
        q = 1
    else:
        p = 1
        q = 0

    result_test = {}
    t0_k = time.time()
    # started from 1 because 0th interval already homogeneous and K+1 because last point is not inclusive
    logging.info('Interval nb %s' % k)
    logging.info('T is: %s' % T)
    n_k = np.int(np.round(n_0 * multiplier ** k))
    n_k_plus1 = np.int(n_0 * multiplier ** (k + 1))

    assert n_k_plus1 > n_k

    if k > 0:
        n_k_minus1 = np.int(np.round(n_0 * multiplier ** (k - 1)))
    else:
        n_k_minus1 = n_0

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

    start_value = starting_values(I_k_plus1, p=p, q=q)  # generate_start_value(garch)
    MLE_I_k_plus1 = mle_estimator(I_k_plus1, start_value, method=solver, bounds=bounds, options={'maxiter': maxiter})

    assert start_k < T - n_k_minus1
    J_k = range(start_k, T - n_k_minus1)
    J_k = np.array(list(J_k))[list(range(0, len(J_k), min_steps))]

    T_k = np.zeros(len(J_k))
    T_k_b = np.empty((len(J_k), num_sim))

    for counter, s in enumerate(J_k):
        logging.info('Break points to go: %s' % str(len(J_k) - counter))

        # New intervals
        A_k = data[start_k_plus1:(s + 1)]
        B_k = data[s + 1:T]

        # Estimator
        start_value = starting_values(A_k, p=p, q=q)
        MLE_A = mle_estimator(A_k, start_value, method=solver, bounds=bounds, options={'maxiter': maxiter})
        start_value = starting_values(B_k, p=p, q=q)
        MLE_B = mle_estimator(B_k, start_value, method=solver, bounds=bounds, options={'maxiter': maxiter})

        # test statistic
        T_k[counter] = -1.0 * (
                MLE_A.fun + MLE_B.fun - MLE_I_k_plus1.fun)  # -1 since we minimized negative log likelihood

        # Bootstrap procedure
        # generating weights
        # Calculating estimators
        # for r in range(num_sim):
        """
        if r % 10 == 0:
            logging.debug('sim %s' % r)
        MLE_A_b, A_success = get_boot_estimator(A_k, weights_A[r, :], max_trial, garch=garch, method=solver,
                                                bounds=bounds, options={'maxiter': maxiter})
        LL_MLE_A_b = - 1.0 * MLE_A_b.fun  # -1 since we minimized negative log likelihood
        if not A_success:
            LL_MLE_A_b = np.nan
            logging.warning('Estimation failed on A for simulation nb %s, message:' % str(r))
            logging.warning(MLE_A_b.message)
            logging.warning('LL value: %s' % str(- np.round(MLE_A_b.fun, 2)))

        MLE_B_b, B_success = get_boot_estimator(B_k, weights_B[r, :], max_trial, garch=garch, method=solver,
                                                bounds=bounds, options={'maxiter': maxiter})
        LL_MLE_B_b = - 1.0 * MLE_B_b.fun  # -1 since we minimized negative log likelihood
        if not B_success:
            LL_MLE_B_b = np.nan
            logging.warning('Estimation failed on B for simulation nb %s' % str(r))
            logging.warning(MLE_B_b.message)
            logging.warning('LL value: %s' % str(- np.round(MLE_B_b.fun, 2)))

        sup_result, sup_success = get_sup_estimator(A_k, weights_A[r, :], B_k, weights_B[r, :], MLE_A.x, MLE_B.x,
                                                    max_trial, garch=garch, method=solver, bounds=bounds,
                                                    options={'maxiter': maxiter})
        LL_sup_result = - 1.0 * sup_result.fun  # -1 since we minimized negative log likelihood
        if not sup_success:
            LL_sup_result = np.nan
            logging.warning('Estimation failed on sup for simulation nb %s' % str(r))
            logging.warning(sup_result.message)
            logging.warning('LL value: %s' % str(- np.round(sup_result.fun, 2)))
        """

        def runner(i):
            return one_bootstrap_test(i, A_k, B_k, MLE_A, MLE_B, max_trial, garch, solver, bounds,
                                       {'maxiter': maxiter}, generate=generate)

        results = Parallel(n_jobs=njobs)(delayed(runner)(i) for i in range(num_sim))

        T_k_b[counter, :] = results

    boot_test = np.nanmax(T_k_b, axis=0)
    logging.info('####  boot_test: %s' % boot_test)
    logging.info('####  T_k: %s' % T_k)
    assert len(boot_test.shape) == 1
    assert boot_test.shape[0] == num_sim

    test_value = max(T_k)
    critical_value = np.sqrt(2.0 * np.quantile(boot_test, level))
    null_is_true = test_value <= critical_value
    logging.info('####  test_value: %s' % test_value)
    logging.info('####  critical_value: %s' % critical_value)
    logging.info('#### NULL REJECTED: %s' % str(not null_is_true))

    start_value = starting_values(I_k, p=p, q=q)  # generate_start_value(garch)
    MLE_I_k = mle_estimator(I_k, start_value, method=solver, bounds=bounds, options={'maxiter': maxiter})
    window = len(I_k)
    scaled_window = len(I_k) / T

    result_test['T_k'] = T_k
    result_test['J_k'] = J_k
    result_test['boot_test'] = boot_test
    result_test['test_value'] = test_value
    result_test['critical_value'] = critical_value
    result_test['H0'] = null_is_true
    result_test['MLE'] = MLE_I_k
    result_test['window'] = window
    result_test['end_k_minus1'] = T - n_k_minus1
    result_test['scaled_window'] = scaled_window

    if save_dir is not None:
        pickle.dump(result_test, open('%s/lpa_result_%s_T_%s.p' % (save_dir, str(k), str(T)), 'wb'))

    t1_k = time.time()
    logging.info('##### TIME IN MIN FOR INTERVAL %s: %s' % (k, str(np.round((t1_k - t0_k) / 60, 2))))

    return result_test, null_is_true, last_test


def test_one_interval_residual(k, data, sigma2, num_sim=2, min_steps=4, multiplier=1.20, n_0=60, level=0.95,
                               max_trial=10, save_dir=None, garch=False, solver='SLSQP', maxiter=100):
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

    if solver == 'SLSQP':
        if garch:
            bounds = get_bounds(data)
        else:
            bounds = get_bounds(data, q=0)
    else:
        bounds = None
    if garch:
        p = 1
        q = 1
    else:
        p = 1
        q = 0

    T = len(data)
    result_test = {k: {}}
    t0_k = time.time()
    # started from 1 because 0th interval already homogeneous and K+1 because last point is not inclusive
    logging.info('Interval nb %s' % k)
    n_k = np.int(np.round(n_0 * multiplier ** k))
    n_k_plus1 = np.int(n_0 * multiplier ** (k + 1))

    assert n_k_plus1 > n_k

    if k > 0:
        n_k_minus1 = np.int(np.round(n_0 * multiplier ** (k - 1)))
    else:
        n_k_minus1 = n_0

    if T - n_k >= 0:
        start_k = T - n_k
    else:
        start_k = 0

    if T - n_k_plus1 >= 0:
        start_k_plus1 = T - n_k_plus1
    else:
        start_k_plus1 = 0

    assert start_k_plus1 < start_k

    I_k = data[start_k:T]
    I_k_plus1 = data[start_k_plus1:T]
    sigma2_I_k_plus1 = sigma2[start_k_plus1:T]

    start_value = starting_values(I_k_plus1, p=p, q=q)  # generate_start_value(garch)
    MLE_I_k_plus1 = mle_estimator(I_k_plus1, start_value, method=solver, bounds=bounds, options={'maxiter': maxiter})

    assert start_k < T - n_k_minus1
    J_k = range(start_k, T - n_k_minus1)
    J_k = np.array(list(J_k))[list(range(0, len(J_k), min_steps))]

    T_k = np.zeros(len(J_k))
    T_k_b = np.empty((len(J_k), num_sim))
    for counter, s in enumerate(J_k):
        logging.info('Break points to go: %s' % str(len(J_k) - counter))
        # New intervals
        A_k = data[start_k_plus1:(s + 1)]
        sigma2_A_k = sigma2[start_k_plus1:(s + 1)]
        B_k = data[s + 1:T]
        sigma2_B_k = sigma2[s + 1:T]
        # Estimator
        start_value = starting_values(A_k, p=p, q=q)
        MLE_A = mle_estimator(A_k, start_value, method=solver, bounds=bounds, options={'maxiter': maxiter})
        start_value = starting_values(B_k, p=p, q=q)
        MLE_B = mle_estimator(B_k, start_value, method=solver, bounds=bounds, options={'maxiter': maxiter})
        # test statistic
        T_k[counter] = -1.0 * (
                MLE_A.fun + MLE_B.fun - MLE_I_k_plus1.fun)  # -1 since we minimized negative log likelihood

        # Bootstrap procedure
        # generating weights
        if garch:
            hatA_k, sigsq_hat, epsA_k = garch_forecast(A_k.reshape(-1), sigma2_A_k, MLE_A.x[0], MLE_A.x[1],
                                                       MLE_A.x[2])
            hatB_k, sigsq_hat, epsB_k = garch_forecast(B_k.reshape(-1), sigma2_B_k, MLE_B.x[0], MLE_B.x[1],
                                                       MLE_B.x[2])
            hatI_k_plus1, sigsq_hat, epsI_k_plus1 = garch_forecast(I_k_plus1.reshape(-1), sigma2_I_k_plus1,
                                                                   MLE_I_k_plus1.x[0], MLE_I_k_plus1.x[1],
                                                                   MLE_I_k_plus1.x[2])
        else:
            hatA_k, sigsq_hat, epsA_k = garch_forecast(A_k.reshape(-1), sigma2_A_k, MLE_A.x[0], MLE_A.x[1])
            hatB_k, sigsq_hat, epsB_k = garch_forecast(B_k.reshape(-1), sigma2_B_k, MLE_B.x[0], MLE_B.x[1])
            hatI_k_plus1, sigsq_hat, epsI_k_plus1 = garch_forecast(I_k_plus1.reshape(-1), sigma2_I_k_plus1,
                                                                   MLE_I_k_plus1.x[0], MLE_I_k_plus1.x[1])

        bootA_k = residual_bootstrap(hatA_k, epsA_k, num_sim=num_sim)
        bootB_k = residual_bootstrap(hatB_k, epsB_k, num_sim=num_sim)
        bootI_k_plus1 = residual_bootstrap(hatI_k_plus1, epsI_k_plus1, num_sim=num_sim)

        # Calculating estimators
        for r in range(num_sim):
            if r % 10 == 0:
                logging.debug('sim %s' % r)
            # A_k
            start_value = starting_values(bootA_k[:, r].reshape(-1, 1), p=p, q=q)  # generate_start_value(garch)
            MLE_A_b = mle_estimator(bootA_k[:, r].reshape(-1, 1), start_value, method=solver, bounds=bounds,
                                    options={'maxiter': maxiter})
            if MLE_A_b.message == "Optimization terminated successfully.":
                A_success = True
            else:
                A_success = False
            LL_MLE_A_b = - 1.0 * MLE_A_b.fun  # -1 since we minimized negative log likelihood
            if not A_success:
                LL_MLE_A_b = np.nan
                logging.warning('Estimation failed on A for simulation nb %s, message:' % str(r))
                logging.warning(MLE_A_b.message)
                logging.warning('LL value: %s' % str(- np.round(MLE_A_b.fun, 2)))

            # B_k
            start_value = starting_values(bootB_k[:, r].reshape(-1, 1), p=p, q=q)  # generate_start_value(garch)
            MLE_B_b = mle_estimator(bootB_k[:, r].reshape(-1, 1), start_value, method=solver, bounds=bounds,
                                    options={'maxiter': maxiter})
            if MLE_B_b.message == "Optimization terminated successfully.":
                B_success = True
            else:
                B_success = False
            LL_MLE_B_b = - 1.0 * MLE_B_b.fun  # -1 since we minimized negative log likelihood
            if not B_success:
                LL_MLE_B_b = np.nan
                logging.warning('Estimation failed on B for simulation nb %s' % str(r))
                logging.warning(MLE_B_b.message)
                logging.warning('LL value: %s' % str(- np.round(MLE_B_b.fun, 2)))

            # I_k_plus1
            start_value = starting_values(bootI_k_plus1[:, r].reshape(-1, 1), p=p, q=q)  # generate_start_value(garch)
            MLE_I_k_plus1_b = mle_estimator(bootI_k_plus1[:, r].reshape(-1, 1), start_value, method=solver,
                                            bounds=bounds, options={'maxiter': maxiter})
            if MLE_B_b.message == "Optimization terminated successfully.":
                I_k_plus1_success = True
            else:
                I_k_plus1_success = False
            LL_MLE_I_k_plus1_b = - 1.0 * MLE_I_k_plus1_b.fun  # -1 since we minimized negative log likelihood
            if not I_k_plus1_success:
                LL_MLE_I_k_plus1_b = np.nan
                logging.warning('Estimation failed on sup for simulation nb %s' % str(r))
                logging.warning(MLE_I_k_plus1_b.message)
                logging.warning('LL value: %s' % str(- np.round(MLE_I_k_plus1_b.fun, 2)))

            T_k_b[counter, r] = LL_MLE_A_b + LL_MLE_B_b - LL_MLE_I_k_plus1_b

    boot_test = np.nanmax(T_k_b, axis=0)
    logging.info('####  boot_test: %s' % boot_test)
    logging.info('####  T_k: %s' % T_k)
    assert len(boot_test.shape) == 1
    assert boot_test.shape[0] == num_sim

    test_value = max(T_k)
    critical_value = np.sqrt(2.0 * np.quantile(boot_test, level))

    logging.info('####  test_value: %s' % test_value)
    logging.info('####  critical_value: %s' % critical_value)
    logging.info('#### NULL REJECTED: %s' % str(test_value > critical_value))

    start_value = starting_values(I_k, p=p, q=q)  # generate_start_value(garch)
    MLE_I_k = mle_estimator(I_k, start_value, method=solver, bounds=bounds, options={'maxiter': maxiter})
    window = len(I_k)
    scaled_window = len(I_k) / T

    result_test[k]['T_k'] = T_k
    result_test[k]['J_k'] = J_k
    result_test[k]['boot_test'] = boot_test
    result_test[k]['test_value'] = test_value
    result_test[k]['critical_value'] = critical_value
    result_test[k]['H0'] = test_value <= critical_value
    result_test[k]['MLE'] = MLE_I_k
    result_test[k]['window'] = window
    result_test[k]['scaled_window'] = scaled_window

    if save_dir is not None:
        pickle.dump(result_test, open('%s/lpa_result_%s.p' % (save_dir, str(k)), 'wb'))

    t1_k = time.time()
    logging.info('##### TIME IN MIN FOR INTERVAL %s: %s' % (k, str(np.round((t1_k - t0_k) / 60, 2))))

    return result_test
