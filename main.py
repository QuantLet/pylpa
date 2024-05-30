
import os, json, time
import pickle as pickle

from pylpa.logger import get_logger, LOGGER

from pylpa.garch.garch import get_max_k, generate_data
from pylpa.lpa import test_one_interval
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
    max_trial = config['max_trial']
    njobs = config['njobs']
    search_every = config['search_every']

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

    data, sigmas = generate_data(omega1=0.2, a1=0.2, b1=0.1, n1=100, omega2=1,
                                 a2=0.2, b2=0.7, n2=50, n0=150)
    data = data.astype('double').reshape(-1,1)
    # Algo
    if "interval_step" not in config:
        if 'K' in config.keys():
            K = config['K']
        else:
            K = get_max_k(MULTIPLIER, N_0, len(data))

        n_ks = [[int(np.round(N_0 * MULTIPLIER ** (k-1))),
                 int(np.round(N_0 * MULTIPLIER ** k)),
                 int(np.round(N_0 * MULTIPLIER ** (k+1)))] for k in range(1, K)]
        cand_windows = [int(np.round(N_0 * MULTIPLIER ** k)) for k in range(1, K + 1)]
    else:
        if len(data) + config["interval_step"] > 2000:
            max_int = 2000 + config["interval_step"]
        else:
            max_int = len(data)
        n_ks = list(range(N_0, max_int, config["interval_step"]))
        n_ks = [[n_ks[i-1], n_ks[i], n_ks[i+1]] for i in range(1, len(n_ks)-1)]
        K = len(n_ks) - 1
        cand_windows = list(range(N_0, max_int, config["interval_step"]))


    if test:
        n_ks = n_ks[:2]

    t1 = time.time()
    if (
        config["model"]['name'] in ['arch', 'garch'] or
        'normalization' in config["model"]["params"] and
        config["model"]["params"]['normalization'] == 'mean_std'
    ):
        LOGGER.info('Centering and reducing to mean 0 and variance 1 on train set')
        # normalize test set with train
        mean_ = np.mean(data)
        std_ = np.std(data)
        train_set = (data - mean_) / std_
        test_obs = (data - mean_) / std_
    else:
        mean_ = None
        std_ = None

    LOGGER.info('Find largest window')
    #cand_windows = [int(np.round(N_0 * MULTIPLIER ** k)) for k in range(1, K + 1)]
    LOGGER.info('Candidate windows: %s' % cand_windows)

    for k in range(len(n_ks)):
        print(k)
        res_k, null_is_true, last_test = test_one_interval(
            k, data, config["model"]["type"], n_ks=n_ks[k],
            T=len(data), num_sim=num_sim,
            min_steps=min_steps, max_trial=max_trial, njobs=njobs,
            solver=config["solver"],  maxiter=config["maxiter"],
            generate=config["generate"],
            **config["model"]["params"]
        )
        if not null_is_true:
            index = res_k['J_k'][np.argmax(res_k['T_k'])]
            LOGGER.info('##### break point detected at index: %s' % str(index))
            break
        else:
            index = np.min(res_k['J_k'])
            assert index == res_k['J_k'][0]
            LOGGER.info('##### No break point detected, use I_k as window with start index %s' % str(index))

