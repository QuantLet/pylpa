import os, json, time
import pdb

from matplotlib import pyplot as plt

from pylpa.logger import get_logger, LOGGER

from pylpa.utils import generate_garch_data
from pylpa.lpa import test_interval
from pylpa.constant import MULTIPLIER, N_0

import numpy as np

from pylpa.models.utils import build_model_from_config
from pylpa.utils import default_config, get_max_k


def find_largest_homogene_interval(data: np.ndarray, **kwargs):
    # Algo
    if "interval_step" not in config:
        if 'K' in config.keys():
            K = config['K']
        else:
            K = get_max_k(MULTIPLIER, N_0, len(data))

        n_ks = [[int(np.round(N_0 * MULTIPLIER ** (k - 1))),
                 int(np.round(N_0 * MULTIPLIER ** k)),
                 int(np.round(N_0 * MULTIPLIER ** (k + 1)))] for k in
                range(1, K)]
    else:
        if len(data) + config["interval_step"] > 2000:
            max_int = 2000 + config["interval_step"]
        else:
            max_int = len(data)
        n_ks = list(range(N_0, max_int, config["interval_step"]))
        n_ks = [[n_ks[i - 1], n_ks[i], n_ks[i + 1]] for i in
                range(1, len(n_ks) - 1)]

    if test:
        n_ks = n_ks[:2]

    if config.get("preprocessing") is not None:
        if config["preprocessing"]["name"] == "StandardScaler":
            LOGGER.info(
                'Centering and reducing to mean 0 and variance 1')
            # normalize test set with train
            mean_ = np.mean(data)
            std_ = np.std(data)
            data = (data - mean_) / std_

    LOGGER.info(f"Create model {config['model']['name']}")
    model = build_model_from_config(config["model"])

    LOGGER.info('Find largest window')
    LOGGER.info('Candidate windows: %s' % n_ks)
    for k in range(len(n_ks)):
        res_k, null_is_true, last_test = test_interval(
            model, k, data, n_ks=n_ks[k],
            T=len(data), num_sim=num_sim,
            min_steps=min_steps, maxtrial=maxtrial, njobs=njobs,
            solver=config["solver"], maxiter=config["maxiter"],
            generate=config["generate"], **kwargs,
        )
        if not null_is_true:
            index = res_k['J_k'][np.argmax(res_k['T_k'])]
            LOGGER.info('Break point detected at index: %s' % str(index))
            LOGGER.info(f"n_ks: {n_ks[k]}")
            return data[-n_ks[k][1]:], index
        else:
            index = np.min(res_k['J_k'])
            assert index == res_k['J_k'][-1]


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
    config = json.load(open('config.json', 'r'))
    if "test" in config.keys():
        test = config["test"]
    else:
        test = False

    if test:
        TEST_SIZE = 2
        LOGGER.info(f'TEST SIZE = {TEST_SIZE}')

    # ALGO PARAMETERS
    num_sim = config['num_sim']
    min_steps = config['min_steps']  # testing point every min_steps
    maxtrial = config['maxtrial']
    njobs = config['njobs']

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
    json.dump(config, open('%s/config.json' % save_dir, 'w'))


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
        interval, index = find_largest_homogene_interval(left)
        intervals.append(interval.tolist())
        breaks.append(index)

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
