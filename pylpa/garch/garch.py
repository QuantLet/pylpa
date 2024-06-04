import numpy as np
import random, scipy
from statsmodels.tsa.arima_model import ARMA
from arch.univariate import ZeroMean, ARCH, GARCH
from functools import partial
from pylpa.garch.constraint import (
    cstr_omega,
    cstr_alpha,
    cstr_beta,
    cstr_alpha_beta
)
from decimal import Decimal

import warnings


# from arch package
def get_bounds(resids, p=1, q=1, power=2, arma=False, mean=False):
    """
    v = np.mean(abs(resids) ** power)
    bounds = [(1e-4, 10.0 * v)]
    bounds.extend([(1e-6, 1.0 - 1e-6)] * p)
    bounds.extend([(1e-6, 1.0 - 1e-6)] * q)
    """

    # http://www.math.pku.edu.cn/teachers/heyb/TimeSeries/lectures/garch.pdf
    # bounds

    Mean = np.mean(resids)
    Var = np.std(resids) ** 2
    S = 1e-6
    if arma:
        assert not mean
        bounds = [(-np.Inf, np.Inf), (-np.Inf, np.Inf), (-np.Inf, np.Inf)]
        bounds.extend([(S, 10 * abs(Var))])
        bounds.extend([(S, 1.0 - S)] * p)
        bounds.extend([(S, 1.0 - S)] * q)
    else:
        if mean:
            bounds = [(-10 * abs(Mean), 10 * abs(Mean))]
        else:
            bounds = [(S, 10 * abs(Var))]
            bounds.extend([(S, 1.0 - S)] * p)
            bounds.extend([(S, 1.0 - S)] * q)

    return bounds


def bounds_check(sigma2, var_bounds):
    DBL_MAX = 10 ^ 6

    if sigma2 < var_bounds[0]:
        sigma2 = var_bounds[0]
    elif sigma2 > var_bounds[1]:
        if sigma2 > DBL_MAX:
            sigma2 = var_bounds[1] + 1000
        else:
            sigma2 = var_bounds[1] + np.log(sigma2 / var_bounds[1])
    return sigma2


def variance_bounds(resids, power=2.0):
    """
    Construct loose bounds for conditional variances.
    These bounds are used in parameter estimation to ensure
    that the log-likelihood does not produce NaN values.
    Parameters
    ----------
    resids : ndarray
        Approximate residuals to use to compute the lower and upper bounds
        on the conditional variance
    power : float, optional
        Power used in the model. 2.0, the default corresponds to standard
        ARCH models that evolve in squares.
    Returns
    -------
    var_bounds : ndarray
        Array containing columns of lower and upper bounds with the same
        number of elements as resids
    """
    nobs = resids.shape[0]

    tau = min(75, nobs)
    w = 0.94 ** np.arange(tau)
    w = w / sum(w)
    var_bound = np.zeros(nobs)
    initial_value = w.dot(resids[:tau] ** 2.0)
    # ewma_recursion(0.94, resids, var_bound, resids.shape[0], initial_value)

    var_bounds = np.vstack((var_bound / 1e6, var_bound * 1e6)).T
    var = resids.var()
    min_upper_bound = 1 + (resids ** 2.0).max()
    lower_bound, upper_bound = var / 1e8, 1e7 * (1 + (resids ** 2.0).max())
    var_bounds[var_bounds[:, 0] < lower_bound, 0] = lower_bound
    var_bounds[var_bounds[:, 1] < min_upper_bound, 1] = min_upper_bound
    var_bounds[var_bounds[:, 1] > upper_bound, 1] = upper_bound

    if power != 2.0:
        var_bounds **= power / 2.0

    return np.ascontiguousarray(var_bounds)


def starting_values(y, garch=True, arma=False):
    # todo: return default value if failed c = mean, a = Yt - mean / Yt-1, etc.
    if arma:
        arma_start_value, arma_resid = arma_starting_values(y, p=1, q=1)
        garch_start_value = garch_starting_values(arma_resid, garch=garch)
        start_value = {'arma': arma_start_value, 'garch': garch_start_value}
    else:
        start_value = garch_starting_values(y, garch=garch)

    return start_value


def arma_starting_values(y, p=1, q=1):
    arma = ARMA(y, (p, q))
    arma = arma.fit(disp=-1)
    """
    try:
    
    except ValueError:
        arma = arma.fit(start_params= [0, 1, 1], disp=-1)
    """

    return list(arma.params), arma.resid


def garch_starting_values(residuals, garch=True):
    am = ZeroMean(residuals)
    if garch:
        am.volatility = GARCH(p=1, q=1)
        res = am.fit(disp='off')
    else:
        am.volatility = ARCH(p=1)
        res = am.fit(disp='off')
    return list(res.params.values)


def get_constraints(p=1, q=1):
    k_arch = p + q
    # alpha[i] >0
    # alpha[i] + gamma[i] > 0 for i<=p, otherwise gamma[i]>0
    # beta[i] >0
    # sum(alpha) + 0.5 sum(gamma) + sum(beta) < 1
    a = np.zeros((k_arch + 2, k_arch + 1))
    for i in range(k_arch + 1):
        a[i, i] = 1.0

    a[k_arch + 1, 1:] = -1.0
    a[k_arch + 1, p + 1: p + 1] = -0.5
    b = np.zeros(k_arch + 2)
    b[k_arch + 1] = -1.0
    return a, b


def get_max_k(multiplier, n_0, T):
    max_k = int(np.log(T / n_0) / np.log(multiplier))
    return max_k


def generate_data(omega1=0.2, a1=0.2, b1=0.1, n1=100, omega2=1, a2=0.2, b2=0.7,
                  n2=50):
    # Simulating a GARCH(1, 1) process
    Y1, sigsq1 = garch_process(n1, omega1, a1, b1)
    Y2, sigsq2 = garch_process(n2, omega2, a2, b2)
    Y = np.concatenate([Y1, Y2, Y1, Y2, Y1, Y2])
    sigsq = np.concatenate([sigsq1, sigsq2, sigsq1, sigsq2, sigsq1, sigsq2])
    breakpoints = [n1, n1 + n2, 2 * n1 + n2, 2 * (n1 + n2), 3 * n1 + 2 * n2,
                   3 * (n1 + n2)]
    return Y, sigsq, breakpoints


def garch_process(n, omega, a1, b1=0, y0=0.001, sigsq0=0.001):
    z = np.random.normal(size=n)
    y = np.zeros_like(z)
    sigsq = np.zeros_like(z)
    sigsq[0] = sigsq0
    y[0] = y0

    for i in range(1, n):
        sigsq[i] = Decimal(omega) + Decimal(a1) * Decimal(
            y[i - 1] ** 2) + Decimal(b1) * Decimal(sigsq[i - 1])
        y[i] = Decimal(z[i]) * Decimal(np.sqrt(sigsq[i]))
    y = np.float_(y)
    sigsq = np.float_(sigsq)

    return y, sigsq


def garch_forecast(y, sigsq, omega, a, b=0):
    assert len(y.shape) == 1
    assert len(sigsq.shape) == 1
    assert len(y) == len(sigsq)
    n = len(y)
    z = np.random.normal(size=n)
    yhat = np.zeros_like(y)
    sigsq_hat = np.zeros_like(y)

    for i in range(1, n):
        sigsq_hat[i] = Decimal(omega) + Decimal(a) * Decimal(
            y[i - 1] ** 2) + Decimal(b) * Decimal(sigsq[i - 1])
        yhat[i] = Decimal(z[i]) * Decimal(np.sqrt(sigsq_hat[i]))
    yhat = np.float_(yhat)
    sigsq_hat = np.float_(sigsq_hat)

    sigsq_hat[0] = np.mean(sigsq_hat)
    yhat[0] = np.sqrt(sigsq_hat[0]) * np.random.normal(size=1)
    residuals = y - yhat

    return yhat, sigsq_hat, residuals


def compute_squared_sigmas(Y, initial_sigma, theta, mu=0.):
    """
    compute the sigmas given the initial guess
    """
    omega = theta[0]
    alpha = theta[1]
    if len(theta) == 2:
        beta = 0
    elif len(theta) == 3:
        beta = theta[2]
    else:
        raise ValueError

    T = len(Y)
    sigma2 = np.ndarray(T)
    # sigma2[0] = Decimal(initial_sigma ** 2)
    var_bounds = variance_bounds(Y)

    prev_sigma = Decimal(initial_sigma ** 2)
    prev_Y = np.std(Y[:, 0])

    for t in range(T):
        ssq = Decimal(omega) + Decimal(alpha) * Decimal(
            (prev_Y - mu) ** 2) + Decimal(beta) * Decimal(prev_sigma)
        sigma2[t] = bounds_check(np.float_(ssq), var_bounds[t])
        prev_Y = Y[t, 0]
        prev_sigma = sigma2[t]

    return sigma2


def compute_residuals_sigmas(Y, initial_sigma, theta):
    """
    compute the residuals and sigmas for arma(1,1)-garch(1,1)
    https://ir.lib.uwo.ca/cgi/viewcontent.cgi?article=2587&context=etd
    """

    c = theta[0]
    a = theta[1]
    b = theta[2]

    omega = theta[3]
    alpha = theta[4]
    if len(theta) == 5:
        beta = 0
    elif len(theta) == 6:
        beta = theta[5]
    else:
        raise ValueError

    T = len(Y)
    sigma2 = np.ndarray(T)
    residuals = np.ndarray(T)
    var_bounds = variance_bounds(Y)

    prev_sigma2 = initial_sigma ** 2
    prev_Y = np.std(Y)
    prev_e = np.std(Y)

    for t in range(T):
        residuals[t] = Y[t, 0] - c - a * prev_Y - b * prev_e
        # warnings.filterwarnings("error")
        try:
            ssq = np.float_(
                Decimal(omega) + Decimal(alpha) * Decimal(
                    prev_e ** 2) + Decimal(beta) * Decimal(prev_sigma2))
        except RuntimeWarning:
            print(t, omega, alpha, prev_e, beta, prev_sigma2)

        sigma2[t] = bounds_check(ssq, var_bounds[t])

        prev_Y = Y[t, 0]
        prev_e = residuals[t]
        prev_sigma2 = sigma2[t]

    return residuals, sigma2


# Likelihood
def garch_loglikelihood(residuals, sigma2):
    return - 0.5 * (
                np.log(2 * np.pi) + np.log(sigma2) + residuals ** 2.0 / sigma2)


def garch_negative_loglikelihood(Y, theta, arma=False, weights=None):
    T = len(Y)
    # Estimate initial sigma squared
    initial_sigma = np.std(Y)  # np.sqrt(np.mean(Y ** 2))

    if not arma:
        # Generate the squared sigma values
        sigma2 = compute_squared_sigmas(Y, initial_sigma, theta)
        residuals = Y.copy()
    else:
        residuals, sigma2 = compute_residuals_sigmas(Y, initial_sigma, theta)

    # print(np.sum([s > 0 for s in sigma2]) - len(sigma2))
    if not np.sum([s > 0 for s in sigma2]) == len(sigma2):
        print(sigma2)
    assert np.sum([s > 0 for s in sigma2]) == len(sigma2)
    # compute
    if weights is None:
        negll = - 1.0 * sum(
            [garch_loglikelihood(residuals[t], sigma2[t]) for t in range(T)])
    else:
        negll = - 1.0 * sum(
            [weights[t] * garch_loglikelihood(residuals[t], sigma2[t]) for t in
             range(T)])

    return negll


def get_start_value(start_theta, A_theta, B_theta, arma=False):
    if not arma:
        assert len(start_theta) == 3 or len(start_theta) == 2

    if 'arma' in start_theta:
        start_theta = start_theta['arma'] + start_theta['garch']

    # arch
    if arma:
        i = 3
    else:
        i = 0
    omega = start_theta[i] - B_theta[i] + A_theta[i]
    alpha = start_theta[i + 1] - B_theta[i + 1] + A_theta[i + 1]

    if len(start_theta) == 3 or len(start_theta) == 6:
        # garch
        beta = start_theta[i + 2] - B_theta[i + 2] + A_theta[i + 2]

    if arma:
        if len(start_theta) == 6:
            start_theta = start_theta[0], start_theta[1], start_theta[
                2], omega, alpha, beta
        elif len(start_theta) == 5:
            start_theta = start_theta[0], start_theta[1], start_theta[
                2], omega, alpha
        else:
            raise ValueError
    else:
        if len(start_theta) == 3:
            start_theta = omega, alpha, beta
        elif len(start_theta) == 2:
            start_theta = omega, alpha
        else:
            raise ValueError

    return start_theta


def generate_start_value(garch=False):
    omega = random.random()
    omega = 0.01 + omega * (1. - 0.01)

    alpha = random.random()
    alpha = 0.25 + alpha * (0.75 - 0.25)

    assert alpha >= 0
    assert omega > 0

    if garch:
        beta = random.random()
        beta = 0.25 + beta * (0.75 - alpha - 0.25)
        assert beta >= 0
        assert alpha + beta <= 1

        return omega, alpha, beta
    else:
        return omega, alpha


# Estimators
def mle_estimator(
        data: np.ndarray, start_value, arma=False, weights=None, **kwargs
):
    """
    data: array type
    start_value: starting value for estimation algo to pass to minimize
    function from scipy
    """

    if not arma:
        assert len(start_value) == 3 or len(start_value) == 2

    if 'arma' in start_value:
        start_value = start_value['arma'] + start_value['garch']

    # Constraints
    if len(start_value) == 2 or len(start_value) == 5:
        # arch
        cons = ({'type': 'ineq', 'fun': partial(cstr_omega, arma)},
                {'type': 'ineq', 'fun': partial(cstr_alpha, arma)})

    elif len(start_value) == 3 or len(start_value) == 6:
        # garch
        cons = ({'type': 'ineq', 'fun': partial(cstr_alpha_beta, arma)},
                {'type': 'ineq', 'fun': partial(cstr_omega, arma)},
                {'type': 'ineq', 'fun': partial(cstr_alpha, arma)},
                {'type': 'ineq', 'fun': partial(cstr_beta, arma)})
    else:
        assert start_value == 5 or start_value == 6

    def objective(theta):
        return garch_negative_loglikelihood(data, theta, arma, weights=weights)

    # warnings.filterwarnings("ignore", category=RuntimeWarning)
    result = scipy.optimize.minimize(
        objective, start_value, constraints=cons, **kwargs
    )

    return result


def sup_estimator(A_data, A_weight, B_data, B_weight, start_value, A_mle,
                  B_mle, arma=False, **kwargs):
    if arma:
        i = 3
    else:
        i = 0

    constant = B_mle - A_mle

    # positive variance
    # omega > 0
    if constant[0 + i] <= 0:
        def sup_cstr_omega(theta):
            return np.array(
                [np.float_(Decimal(theta[0 + i]) + Decimal(constant[0 + i]))])
    else:
        def sup_cstr_omega(theta):
            return np.array([np.float_(Decimal(theta[0 + i]))])
    # alpha >= 0
    if constant[1 + i] <= 0:
        def sup_cstr_alpha(theta):
            return np.array(
                [np.float_(Decimal(theta[1 + i]) + Decimal(constant[1 + i]))])
    else:
        def sup_cstr_alpha(theta):
            return np.array([theta[1 + i]])

    # arch
    if len(constant) == 2 or len(constant) == 5:
        cons_sup = ({'type': 'ineq', 'fun': sup_cstr_omega},
                    {'type': 'ineq', 'fun': sup_cstr_alpha})
    # garch
    elif len(constant) == 3 or len(constant) == 6:
        # beta >= 0
        if constant[2 + i] <= 0:
            def sup_cstr_beta(theta):
                return np.array([np.float_(
                    Decimal(theta[2 + i]) + Decimal(constant[2 + i]))])
        else:
            def sup_cstr_beta(theta):
                return np.array([theta[2 + i]])

        # stationarity
        # alpha + beta <= 1
        c_alpha_beta = B_mle[1 + i] - A_mle[1 + i] + B_mle[2 + i] - A_mle[
            2 + i]
        if c_alpha_beta >= 0:
            def sup_cstr_alpha_beta(theta):
                return np.array(
                    [np.float_(Decimal(1.) - (Decimal(theta[1 + i]) + Decimal(
                        theta[2 + i]) - Decimal(c_alpha_beta)))])
        else:
            def sup_cstr_alpha_beta(theta):
                return np.array([np.float_(Decimal(1.) - (
                            Decimal(theta[1 + i]) + Decimal(theta[2 + i])))])

        cons_sup = ({'type': 'ineq', 'fun': sup_cstr_alpha_beta},
                    {'type': 'ineq', 'fun': sup_cstr_omega},
                    {'type': 'ineq', 'fun': sup_cstr_alpha},
                    {'type': 'ineq', 'fun': sup_cstr_beta})
    else:
        raise ValueError

    def objective(theta):
        bias_theta = theta + B_mle - A_mle
        return (
                garch_negative_loglikelihood(A_data, theta, arma,
                                             weights=A_weight) +
                garch_negative_loglikelihood(
                    B_data, bias_theta, arma, weights=B_weight)
        )

    result = scipy.optimize.minimize(objective,
                                     start_value,
                                     constraints=cons_sup,
                                     **kwargs)

    return result
