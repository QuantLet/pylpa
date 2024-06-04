from statsmodels.tsa.arima_model import ARMA


def starting_values(y, p=1, q=1):
    arma = ARMA(y, (p, q))
    arma = arma.fit(disp=-1)

    return list(arma.params), arma.resid
