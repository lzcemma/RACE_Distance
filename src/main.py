import numpy as np
import argparse
import torch
import random

import algo_fedavg
import algo_sketch_sample
import algo_personalization

DATASETS = [
    "mnist_dirichlet_uniform",
]


def parse_args():
    default_K = 3  # number of devices to train each round
    default_T = 200  # number of rounds
    default_E = 20  # number of epochs for all devices
    default_N = 5  # number of cold start client
    default_batch_size = 32  # batch size for all devices
    default_learn_rate = 0.0001  # learning rate for all devices
    default_seed = 0  # random seed
    # RACE parameters
    default_raceK = 6  # number of hash functions
    default_raceR = 1000  # number of rows

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    req_grp = parser.add_argument_group(title="required arguments")
    req_grp.add_argument(
        "--algo",
        choices=[
            "fedavg",
            "sample",
            "personalize",
        ],
        required=True,
        help="algorithm to run",
    )
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument(
        "--K",
        type=int,
        default=default_K,
        help="number of devices to train each round (default: {:d})".format(default_K),
    )
    parser.add_argument(
        "--T",
        type=int,
        default=default_T,
        help="number of rounds (default: {:d})".format(default_T),
    )
    parser.add_argument(
        "--E",
        type=int,
        default=default_E,
        help="number of epochs for all devices (default: {:d})".format(default_E),
    )
    parser.add_argument(
        "--N",
        type=int,
        default=default_N,
        help="number of cold start client)".format(default_N),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_batch_size,
        help="batch size for all devices (default: {:d})".format(default_batch_size),
    )
    parser.add_argument(
        "--learn_rate",
        type=float,
        default=default_learn_rate,
        help="learning rate for all devices (default: {:.4f})".format(
            default_learn_rate
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="learning rate for all devices (default: {:.4f})".format(
            default_learn_rate
        ),
    )

    parser.add_argument(
        "--raceK",
        type=int,
        default=default_raceK,
        help="number of devices to train each round (default: {:d})".format(default_K),
    )
    parser.add_argument(
        "--raceR",
        type=int,
        default=default_raceR,
        help="number of rounds (default: {:d})".format(default_T),
    )

    parser.add_argument(
        "--leaf", action="store_true", help="use LEAF data (default: false)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed,
        help="random seed (default: {:d})".format(default_seed),
    )

    parser.add_argument(
        "--tensorboard", action="store_true", help="Plot with tensorboard"
    )
    parser.add_argument(
        "--save_model", action="store_true", help="Save model checkpoints"
    )
    parser.add_argument(
        "--train_global", action="store_true", help="Plot with tensorboard"
    )
    parser.add_argument(
        "--train_local", action="store_true", help="Train model for each personlized device, instead of loading"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.algo == "fedavg":
        algo_fedavg.fed_avg(args)
    elif args.algo == "sample":
        algo_sketch_sample.fed_race_sample_prob(args)
    elif args.algo == "personalize":
        algo_personalization.fed_race_personalize(args)


#
#
#
