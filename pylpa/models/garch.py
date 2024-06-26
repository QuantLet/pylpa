from decimal import Decimal

import numpy as np
import arch

from pylpa.models.arch import ARCH, variance_bounds, bounds_check


def starting_values(y, p, q):
    am = arch.univariate.ZeroMean(y)
    am.volatility = arch.univariate.GARCH(p=p, q=q)
    res = am.fit(disp='off')
    return list(res.params.values)


class GARCH(ARCH):
    """

    :param p: ARCH p order
    :param q: GARCH q order
    """
    def __init__(
            self, p: int = 1, q: int = 1, dist: str = "normal"
    ):
        super().__init__(p=p, dist=dist)
        assert q > 0, "q must be larger than 1"
        if p + q > 2:
            raise NotImplementedError("For now only GARCH(1,1) is implemented")
        self.q = q
        self.params_constraints = (
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
        self.sup_constraints = (
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

        if len(constant) == 3 or len(constant) == 6:
            # beta >= 0
            if constant[2] <= 0:
                def sup_cstr_beta(theta):
                    return np.array([np.float_(
                        Decimal(theta[2]) + Decimal(constant[2]))])
            else:
                def sup_cstr_beta(theta):
                    return np.array([theta[2]])

            # stationarity
            # alpha + beta <= 1
            c_alpha_beta = B_mle[1] - A_mle[1] + B_mle[2] - A_mle[2]
            if c_alpha_beta >= 0:
                def sup_cstr_alpha_beta(theta):
                    return np.array(
                        [np.float_(Decimal(1.) - (Decimal(theta[1]) + Decimal(
                            theta[2]) - Decimal(c_alpha_beta)))])
            else:
                def sup_cstr_alpha_beta(theta):
                    return np.array([np.float_(Decimal(1.) - (
                                Decimal(theta[1]) + Decimal(theta[2])))])

            cons_sup = ({'type': 'ineq', 'fun': sup_cstr_alpha_beta},
                        {'type': 'ineq', 'fun': sup_cstr_omega},
                        {'type': 'ineq', 'fun': sup_cstr_alpha},
                        {'type': 'ineq', 'fun': sup_cstr_beta})
        else:
            raise ValueError

        return cons_sup

    def negative_loglikelihood(self, y, theta, weights=None):
        if self.dist == "normal":
            loglikelihood = self.normal_loglikelihood
        else:
            raise NotImplementedError(self.dist)

        T = len(y)
        # Generate the process values
        y, fitted_values = self.compute_fitted_values(y, np.var(y), theta)

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

    @staticmethod
    def compute_fitted_values(y, y_hat_0, theta, mu=0.):
        """

        :param y:
        :param y_hat_0:
        :param theta:
        :param mu:
        :return:
        """
        if len(theta) > 3:
            raise NotImplementedError

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