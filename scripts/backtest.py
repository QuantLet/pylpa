import os, json, time
import datetime as dt

import pandas as pd
from arch.univariate import arch_model

from pylpa.data import get_returns_from_prices
from pylpa.logger import get_logger, LOGGER

from pylpa.lpa import find_largest_homogene_interval
from pylpa.constant import MULTIPLIER, N_0

import numpy as np

from pylpa.utils import default_config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            prog="Backtest",
            description="Run the LPA",
        )
    parser.add_argument(
        "--level",
        type=str,
        help="Logger level",
        default="info",
    )
    parser.add_argument(
        "--quantiles",
        help="VaR levels",
        default=[0.01, 0.025, 0.05],
    )
    args = parser.parse_args()

    # CONFIG ALGO
    config = json.load(open('scripts/backtest_config.json', 'r'))
    if "test" in config.keys():
        test = config["test"]
    else:
        test = False

    if test:
        TEST_SIZE = 2
        LOGGER.info(f'TEST SIZE = {TEST_SIZE}')


    if 'seed' in config.keys():
        np.random.seed(seed=config['seed'])

    config = default_config(config)

    # Save directory
    save_dir = config.get("save_dir", "saved_result")
    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_dir}_{now}"

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    else:
        LOGGER.warning("Overwriting save_dir")
    LOGGER = get_logger(
        "LPA-main", level=args.level, save_path=f"{save_dir}/run.log"
    )

    # dump config
    json.dump(config, open('%s/config.json' % save_dir, 'w'))

    tc1 = time.time()
    fit_window = {}

    dates, returns = get_returns_from_prices(
        path=config["data"]["path"], feature_name=config["data"]["feature"]
    )
    min_size = int(np.round(N_0 * MULTIPLIER ** 2))
    # Get previous N_k
    # If len(window) < previous N_k do not perform test and take last
    # breakpoint
    # Otherwise perform test
    # Fit GARCH on interval
    # Predict
    indices = list(range(min_size, len(returns)))
    value_at_risk = np.zeros((len(indices), len(args.quantiles)))
    breakpoints = []
    c = 0
    for i in indices:
        window = returns[:i,:]
        intervals = []
        breaks = []
        if config["data"].get("preprocessing") is not None:
            if config["data"]["preprocessing"]["name"] == "StandardScaler":
                LOGGER.info(
                    'Centering and reducing to mean 0 and variance 1')
                # normalize test set with train
                mean_ = np.mean(window)
                std_ = np.std(window)
                window = (window - mean_) / std_
            else:
                raise NotImplementedError(config["preprocessing"])
        else:
            window = window

        interval, index = find_largest_homogene_interval(
            window, config["model"], K=config["K"],
            interval_step=config["interval_step"],
            min_steps=config['min_steps'],
            solver=config['solver'], maxiter=config['maxiter'],
            maxtrial=config['maxtrial'],
            generate=config["bootstrap"]["generate"],
            num_sim=config["bootstrap"]['num_sim'],
            njobs=config["bootstrap"]['njobs']
        )
        if index != -1:
            breakpoints.append([dates[index], index])

        # Fit garch on interval
        am = arch_model(
            interval,
            mean="AR" if config["model"]["name"] == "ARMAGARCH" else "Zero",
            dist="normal", **config["model"]["params"]
        )
        res = am.fit()
        forecasts = res.forecast(horizon=1)
        cond_mean = forecasts.mean
        cond_var = forecasts.variance
        q = am.distribution.ppf(args.quantiles)
        VaR = -cond_mean.values - np.sqrt(cond_var).values * q[None, :]
        if config["data"].get("preprocessing") is not None:
            if config["data"]["preprocessing"]["name"] == "StandardScaler":
                VaR *= std_
                VaR += mean_
            else:
                raise NotImplementedError(config["preprocessing"])
        value_at_risk[c, :] = VaR
        if c % 10 == 0:
            result = pd.DataFrame(
                value_at_risk[:c,:],
                columns=[f"q_{q}" for q in args.quantiles],
                index=dates[min_size:i],
            )
            result.to_csv(f"{save_dir}/res_{c}.csv")
            pd.DataFrame(
                breakpoints, columns=["dates", "index"]
            ).to_csv(f"{save_dir}/res_breakpoints_{c}.csv", index=False)
        c += 1

    # Save forecasts
    result = pd.DataFrame(
        value_at_risk, columns=[f"q_{q}" for q in args.quantiles],
        index=dates[min_size:c],
    )
    result.to_csv(f"{save_dir}/results.csv")
    pd.DataFrame(
        breakpoints, columns=["dates", "index"]
    ).to_csv(f"{save_dir}/breakpoints.csv", index=False)

    # Clean temp files
    for i in range(0, c + 1, 10):
        os.remove(f"{save_dir}/res_{i}.csv")
        os.remove(f"{save_dir}/res_breakpoints_{i}.csv")
