import numpy as np
import scipy
from decimal import Decimal

import warnings


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
                negative_loglikelihood(A_data, theta, arma,
                                             weights=A_weight) +
                negative_loglikelihood(
                    B_data, bias_theta, arma, weights=B_weight)
        )

    result = scipy.optimize.minimize(objective,
                                     start_value,
                                     constraints=cons_sup,
                                     **kwargs)

    return result
