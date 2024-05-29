import numpy as np
from decimal import Decimal

# Define the constraints for our minimizer
# stationarity
def cstr_alpha_beta(arma, theta):
    if arma:
        return np.array([np.float_(Decimal(1.) - Decimal(theta[4] + theta[5]))])
    else:
        return np.array([np.float_(Decimal(1.) - Decimal(theta[1] + theta[2]))])


# positive variance
def cstr_omega(arma, theta):
    if arma:
        return np.array([np.float_(Decimal(theta[3]) - Decimal(1e-6))])
    else:
        return np.array([np.float_(Decimal(theta[0]) - Decimal(1e-6))])


def cstr_alpha(arma, theta):
    if arma:
        return np.array([theta[4]])
    else:
        return np.array([theta[1]])


def cstr_beta(arma, theta):
    if arma:
        return np.array([theta[5]])
    else:
        return np.array([theta[2]])