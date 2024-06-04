from decimal import Decimal

import numpy as np
import arch


def starting_values(y, p):
    am = arch.ZeroMean(y)
    am.volatility = arch.ARCH(p=p)
    res = am.fit(disp='off')
    return list(res.params.values)


class ARCH:
    """

    :param p: ARCH p order
    :param q: GARCH q order
    """
    def __init__(
            self, p: int = 1, q: int = 1, mean: bool = False
    ):
        self.p = p
        self.q = q
        self.mean = mean
        self.constraints = (
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
            bounds.extend([(S, 1.0 - S)] * self.q)

        return bounds

    def get_starting_values(self, y):
        return starting_values(y, self.p, self.q)

    @property
    def constraints(self):
        return self.constraints
