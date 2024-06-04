from decimal import Decimal

import numpy as np
import arch

from pylpa.logger import LOGGER


def starting_values(y, p, q):
    am = arch.univariate.ZeroMean(y)
    am.volatility = arch.univariate.GARCH(p=p, q=q)
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


class GARCH:
    """

    :param p: ARCH p order
    :param q: GARCH q order
    """
    def __init__(
            self, p: int = 1, q: int = 1, dist: str = "normal"
    ):
        self.p = p
        self.q = q
        self.dist = dist
        self.constraints = (
            # stationary: \alpha + \beta < 1
            {
                'type': 'ineq',
                'fun': lambda x: np.array([
                    np.float_(Decimal(1.) - Decimal(x[1] + x[2]))
                ])
            },
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
            # \beta > 0
            {
                'type': 'ineq',
                'fun': lambda x: np.array([x[2]])
            }
        )
        self._initial_params = None

    def get_bounds(self, y):
        """
        source: http://www.math.pku.edu.cn/teachers/heyb/TimeSeries/lectures/garch.pdf

        :param y:
        :return:
        """
        Var = np.std(y) ** 2
        S = 1e-6

        bounds = [(S, 10 * abs(Var))]
        bounds.extend([(S, 1.0 - S)] * self.p)
        bounds.extend([(S, 1.0 - S)] * self.q)

        return bounds

    # @property
    # def initial_params(self):
    #     return self._initial_params
    #
    # @initial_params.setter
    # def initial_params(self, initial_params):
    #     self._initial_params = initial_params

    def estimate_initial_params(self, y):
        return starting_values(y, self.p, self.q)

    @staticmethod
    def compute_fitted_values(y, y_hat_0, theta, mu=0.):
        """

        :param y:
        :param y_hat_0:
        :param theta:
        :param mu:
        :return:
        """
        if np.mean(y) > 1e-3:
            LOGGER.warning("Sample does not have a 0 mean. Overwritting with sample mean")
            mu = np.mean(y)

        assert len(theta) == 3
        omega = theta[0]
        alpha = theta[1]
        beta = theta[2]

        T = len(y)
        y_hat = np.ndarray(T)
        var_bounds = variance_bounds(y)
        prev_value = y_hat_0
        prev_y = np.std(y[:, 0])

        for t in range(T):
            ssq = Decimal(omega) + Decimal(alpha) * Decimal(
                (prev_y - mu) ** 2) + Decimal(beta) * Decimal(prev_value)
            y_hat[t] = bounds_check(np.float_(ssq), var_bounds[t])
            prev_y = y[t, 0]
            prev_value = y_hat[t]

        return y, y_hat
