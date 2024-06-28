import os, json, time

from matplotlib import pyplot as plt

from pylpa.logger import get_logger, LOGGER

from pylpa.utils import generate_garch_data
from pylpa.lpa import find_largest_homogene_interval
from pylpa.constant import MULTIPLIER, N_0

import numpy as np

from pylpa.utils import default_config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            prog="LPA",
            description="Run the LPA",
        )
    parser.add_argument(
        "--level",
        type=str,
        help="Logger level",
        default="info",
    )
    args = parser.parse_args()

    # CONFIG ALGO
    config = json.load(open('scripts/demo_config.json', 'r'))
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

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    else:
        LOGGER.warning("Overwriting save_dir")
    LOGGER = get_logger(
        "LPA-main", level=args.level, save_path=f"{save_dir}/run.log"
    )

    # dump config
    json.dump(config, open('%s/demo_config.json' % save_dir, 'w'))

    tc1 = time.time()
    fit_window = {}

    data, sigmas, gen_breakpoints = generate_garch_data(
        omega1=0.2, a1=0.2, b1=0.1, n1=100, omega2=1, a2=0.2, b2=0.7, n2=70,
    )
    LOGGER.info(f"Breakpoints: {gen_breakpoints}")
    data = data.astype('double').reshape(-1,1)


    left = data.copy()
    intervals = []
    breaks = []
    while len(left) > N_0*MULTIPLIER**2:
        if config["data"].get("preprocessing") is not None:
            if config["data"]["preprocessing"]["name"] == "StandardScaler":
                LOGGER.info(
                    'Centering and reducing to mean 0 and variance 1')
                # normalize test set with train
                mean_ = np.mean(left)
                std_ = np.std(left)
                window = (left - mean_) / std_
            else:
                raise NotImplementedError(config["preprocessing"])
        else:
            window = left

        interval_step = config.get("interval_step")
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
        intervals.append(interval.tolist())
        breaks.append(index)

        if index != -1:
            plt.plot(left)
            plt.vlines(index, min(left), max(left), colors="red")
            plt.savefig(
                f"{save_dir}/plot_{index}.png", transparent=True,
                bbox_inches="tight"
            )
            plt.close()
        left = left[:-len(interval)]
        json.dump(intervals, open(f"{save_dir}/intervals.json", "w"))
        json.dump(breaks, open(f"{save_dir}/breakpoints.json", "w"))
