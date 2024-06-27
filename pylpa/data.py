import pdb

import numpy as np
import pandas as pd

from pylpa.logger import LOGGER


def get_returns_from_prices(path: str, feature_name: str = "log_returns"):
    """
    The excel table must contain two columns with names ["Date", "Close"]
    :param path: path to the file
    :param feature_name: log_returns or returns
    :return:
    """
    ext = path.split(".")[-1]
    if ext == "xlsx":
        try:
            prices = pd.read_excel(path)
        except ImportError as _exc:
            LOGGER.error("You must install 'openpyxl'!")
            raise _exc
    else:
        raise NotImplementedError(ext)

    dates = pd.to_datetime(prices["Date"]).tolist()[1:]
    prices = prices["Close"].values

    if feature_name == "returns":
        returns = prices[1:]/prices[:-1] - 1
    elif feature_name == "log_returns":
        returns = np.log(prices[1:] / prices[:-1])
    else:
        raise NotImplementedError(feature_name)

    mask = ~np.isnan(returns)

    dates = np.array(dates)[mask]
    returns = returns[mask].reshape(-1, 1)

    return dates, returns.astype("double")