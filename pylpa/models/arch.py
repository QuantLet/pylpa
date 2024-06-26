from decimal import Decimal

import numpy as np
import arch

from pylpa.logger import LOGGER


def starting_values(y, p):
    am = arch.univariate.ZeroMean(y)
    am.volatility = arch.univariate.ARCH(p=p)
    res = am.fit(disp='off')
    return list(res.params.values)


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


class ARCH:
    """

    :param p: ARCH p order
    """
    def __init__(
            self, p: int = 1, mean: bool = False, dist: str = "normal"
    ):
        assert p > 0, "p must be larger than 0"
        self.p = p
        self.mean = mean
        self.dist = dist
        self.params_constraints = (
            # positive variance: \omega > 0
            {
                'type': 'ineq',
                'fun': lambda x: np.array([
                    np.float_(Decimal(x[0]) - Decimal(1e-6))
                ])
            },
            # \alpha > 0
            {
                'type': 'ineq',
                'fun': lambda x: np.array([x[1]])
            },
        )

    def estimate_initial_params(self, y):
        return starting_values(y, self.p)

    def get_bounds(self, y):
        """
        source: http://www.math.pku.edu.cn/teachers/heyb/TimeSeries/lectures/garch.pdf

        :param y:
        :return:
        """
        Mean = np.mean(y)
        Var = np.std(y) ** 2
        S = 1e-6

        if self.mean:
            bounds = [(-10 * abs(Mean), 10 * abs(Mean))]
        else:
            bounds = [(S, 10 * abs(Var))]
            bounds.extend([(S, 1.0 - S)] * self.p)

        return bounds

    def get_starting_values(self, y):
        return starting_values(y, self.p)

    @staticmethod
    def normal_loglikelihood(resid, sigma2):
        return - 0.5 * (
                np.log(2 * np.pi) + np.log(sigma2) + resid ** 2.0 / sigma2
        )

    def negative_loglikelihood(self, y, theta, weights=None):
        if self.dist == "normal":
            loglikelihood = self.normal_loglikelihood
        else:
            raise NotImplementedError(self.dist)

        T = len(y)
        # Generate the process values
        y, fitted_values = self.compute_fitted_values(y, theta)

        assert all([s > 0 for s in fitted_values])
        # compute
        if weights is None:
            negll = - 1.0 * sum([
                loglikelihood(y[t], fitted_values[t]) for t in range(T)
            ])
        else:
            negll = - 1.0 * sum([
                weights[t] * loglikelihood(y[t], fitted_values[t])
                for t in range(T)
            ])
        return negll

    @property
    def constraints(self):
        return self.constraints

    @staticmethod
    def compute_fitted_values(y, theta, mu=0.):
        """

        :param y:
        :param y_hat_0:
        :param theta:
        :param mu:
        :return:
        """
        # if np.mean(y) > 1e-3:
        #     LOGGER.warning(
        #         "Sample does not have a 0 mean. Overwritting with sample mean"
        #     )
        #     mu = np.mean(y)

        assert len(theta) == 2
        omega = theta[0]
        alpha = theta[1]

        T = len(y)
        y_hat = np.ndarray(T)
        var_bounds = variance_bounds(y)
        prev_y = np.std(y[:, 0])

        for t in range(T):
            ssq = Decimal(omega) + Decimal(alpha) * Decimal((prev_y - mu) ** 2)
            y_hat[t] = bounds_check(np.float_(ssq), var_bounds[t])
            prev_y = y[t, 0]

        return y, y_hat

    @staticmethod
    def constraint_sup(A_mle, B_mle):
        constant = B_mle - A_mle
        # positive variance
        # omega > 0
        if constant[0] <= 0:
            def sup_cstr_omega(theta):
                return np.array(
                    [np.float_(Decimal(theta[0]) + Decimal(constant[0]))])
        else:
            def sup_cstr_omega(theta):
                return np.array([np.float_(Decimal(theta[0]))])
        # alpha >= 0
        if constant[1] <= 0:
            def sup_cstr_alpha(theta):
                return np.array(
                    [np.float_(Decimal(theta[1]) + Decimal(constant[1]))])
        else:
            def sup_cstr_alpha(theta):
                return np.array([theta[1]])
        if len(constant) == 2 or len(constant) == 5:
            cons_sup = ({'type': 'ineq', 'fun': sup_cstr_omega},
                        {'type': 'ineq', 'fun': sup_cstr_alpha})

        return cons_sup
