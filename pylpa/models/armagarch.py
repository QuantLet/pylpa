from decimal import Decimal

import numpy as np

from pylpa.models.arma import starting_values as arma_starting_values
from pylpa.models.garch import starting_values


class ARMAGARCH:
    """

    :param p: ARMA p order
    :param q: ARMA q order
    :param P: ARCH P order
    :param Q: GARCH Q order
    """
    def __init__(
            self, y, p=1, q=1, P=1, Q=1,
    ):
        self.p = p
        self.q = q
        self.P = P
        self.Q = Q
        self.constraints = (
            # stationary: \alpha + \beta < 1
            {
                'type': 'ineq',
                'fun': lambda x: np.array([
                    np.float_(Decimal(1.) - Decimal(x[4] + x[5]))
                ])
            },
            # positive variance: \omega > 0
            {
                'type': 'ineq',
                'fun': lambda x: np.array([
                    np.float_(Decimal(x[3]) - Decimal(1e-6))
                ])
            },
            # \alpha > 0
            {
                'type': 'ineq',
                'fun': lambda x: np.array([x[4]])
            },
            # \beta > 0
            {
                'type': 'ineq',
                'fun': lambda x: np.array([x[5]])
            }
        )

    def bounds(self, y):
        """
        source: http://www.math.pku.edu.cn/teachers/heyb/TimeSeries/lectures/garch.pdf

        :param y:
        :return:
        """
        Var = np.std(y) ** 2
        S = 1e-6
        bounds = [(-np.Inf, np.Inf), (-np.Inf, np.Inf),
                  (-np.Inf, np.Inf)]
        bounds.extend([(S, 10 * abs(Var))])
        bounds.extend([(S, 1.0 - S)] * self.P)
        bounds.extend([(S, 1.0 - S)] * self.Q)

        return bounds

    def starting_values(self, y):
        arma_start_value, arma_resid = arma_starting_values(
            y, p=self.p, q=self.q
        )
        garch_start_value = starting_values(
            arma_resid, p=self.P, q=self.Q)
        # start_value = {'arma': arma_start_value, 'garch': garch_start_value}

        return arma_start_value + garch_start_value
