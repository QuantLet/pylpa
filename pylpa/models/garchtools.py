import numpy as np
from decimal import Decimal


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

