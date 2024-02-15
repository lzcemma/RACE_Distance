"""
@file: generate_data.py
@author: Zichang Liu
@notes:
	* Support 4 data split 
	[1] homo: uniform data split
	[2] dirichlet: partition data to reflect Label distribution skew following https://github.com/IBM/probabilistic-federated-neural-matching
    [3] dirichlet_uniform: Combination of homo and dirichlet
	[3] byclass: specify how many class is allowed at each device
"""

import numpy as np
from torchvision.datasets import MNIST

import argparse

import json

import random

from torchvision import transforms


def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train_ds = MNIST(datadir, train=True, transform=transform, download=True)
    mnist_test_ds = MNIST(datadir, train=False, transform=transform, download=True)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.targets
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.targets

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, args):
    print(f" # devices {args.n_devices}; Partition {args.partition}")
    if dataset == "mnist":
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    else:
        raise NotImplementedError

    n_classes = len(set(y_train.tolist()))
    if args.partition == "homo":
        n_train = X_train.shape[0]
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, args.n_devices)
        net_dataidx_map = {i: batch_idxs[i].tolist() for i in range(args.n_devices)}

        n_test = X_test.shape[0]
        idxs = np.random.permutation(n_test)
        batch_idxs = np.array_split(idxs, args.n_devices)
        net_dataidx_map_test = {
            i: batch_idxs[i].tolist() for i in range(args.n_devices)
        }

    elif args.partition == "dirichlet":
        print(f"Dirichlet alpha {args.alpha}")
        path_str = f"{args.partition}_{args.n_devices}_{args.alpha}_{args.minsize}"
        n_train = X_train.shape[0]
        print(f"# of train: {n_train}")
        min_size = 0
        K = 10
        N = y_train.shape[0]
        net_dataidx_map_test = None
        net_dataidx_map = {}
        while min_size < args.minsize:
            idx_batch = [[] for _ in range(args.n_devices)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.alpha, args.n_devices))
                ## Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / args.n_devices)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(args.n_devices):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif args.partition == "byclass":
        path_str = f"{args.partition}_neighbor{args.neighbor}_{args.n_devices}_{args.n_classes_per_device}"
        print(f"# class per device {args.n_classes_per_device}")
        train_dict = {}
        for idx, c_i in enumerate(y_train):
            if c_i.item() in train_dict:
                train_dict[c_i.item()].append(idx)
            else:
                train_dict[c_i.item()] = [idx]

        test_dict = {}
        for idx, c_i in enumerate(y_test):
            if c_i.item() in test_dict:
                test_dict[c_i.item()].append(idx)
            else:
                test_dict[c_i.item()] = [idx]

        _min_sample = args.minsize
        _mean = 0
        _sigma = 2.0

        n_samples_per_device = (
            np.random.lognormal(_mean, _sigma, args.n_devices).astype(int) + _min_sample
        )
        net_dataidx_map = {}
        net_dataidx_map_test = {}
        _train_pct = 0.9
        class_choice = []
        if args.neighbor:
            for j in range(int(0.5 * args.n_devices)):
                class_choice += [
                    np.random.choice(
                        n_classes, args.n_classes_per_device, replace=False
                    ).tolist()
                ]
            class_choice = np.array(class_choice)
            class_choice = np.concatenate([class_choice, class_choice], axis=0)

        for dev_id, n_sample_i in enumerate(n_samples_per_device):
            n_train = int(_train_pct * n_sample_i)
            n_test = n_sample_i - n_train
            # use uniform sample per class for now
            train_samples_per_class = [
                n_train // args.n_classes_per_device
                + (1 if i < n_train % args.n_classes_per_device else 0)
                for i in range(args.n_classes_per_device)
            ]
            test_samples_per_class = [
                n_test // args.n_classes_per_device
                + (1 if i < n_test % args.n_classes_per_device else 0)
                for i in range(args.n_classes_per_device)
            ]
            # sample classes
            if args.neighbor:
                classes = class_choice[dev_id]
            else:
                classes = np.random.choice(
                    n_classes,
                    args.n_classes_per_device,
                    replace=False,
                )

            train_idxs = []
            test_idxs = []
            for c_i, s_i in zip(classes, train_samples_per_class):
                train_idxs_ci = np.random.choice(
                    train_dict[c_i], s_i, replace=False
                )  # note: devices may have overlapping samples
                train_idxs.extend(train_idxs_ci.tolist())
            random.shuffle(train_idxs)
            net_dataidx_map[dev_id] = train_idxs

            for c_i, s_i in zip(classes, test_samples_per_class):
                test_idxs_ci = np.random.choice(
                    test_dict[c_i], s_i, replace=False
                )  # note: devices may have overlapping samples

                test_idxs.extend(test_idxs_ci.tolist())
            random.shuffle(test_idxs)
            net_dataidx_map_test[dev_id] = test_idxs

    elif args.partition == "dirichlet_uniform":
        path_str = f"{args.partition}_{args.n_devices}_{args.ratio}_{args.alpha}_{args.minsize}"
        n_dir_device = int(args.ratio * args.n_devices)
        n_uni_device = args.n_devices - n_dir_device
        print(f"# Dirchlet device: {n_dir_device}, # Uniform device: {n_uni_device}")
        print(f"Dirichlet alpha {args.alpha}")
        n_train = X_train.shape[0]
        min_size = 0
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        idxs = np.random.permutation(n_train)
        dir_idx, uni_idx = (
            idxs[: int(n_train * args.ratio)],
            idxs[int(n_train * args.ratio) :],
        )

        while min_size < args.minsize:
            idx_batch = [[] for _ in range(n_dir_device)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.alpha, n_dir_device))
                ## Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_dir_device)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_dir_device):
            device_idx = []
            for i in idx_batch[j]:
                if i not in uni_idx:
                    device_idx += [i]

            np.random.shuffle(device_idx)
            net_dataidx_map[j] = device_idx

        batch_idxs = np.array_split(uni_idx, n_uni_device)
        for i in range(n_dir_device, args.n_devices):
            net_dataidx_map[i] = batch_idxs[i - n_dir_device].tolist()
        net_dataidx_map_test = None

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        net_dataidx_map_test,
        path_str,
    )


def parse_args():
    default_n_devices = 30
    default_seed = 0
    default_alpha = 0.5
    default_n_classes_per_device = 2
    default_ratio = 0.5
    default_partition = ["dirichlet"]
    partition_choices = ["dirichlet", "homo", "byclass", "dirichlet_uniform"]
    dataset_choices = ["mnist"]
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--n_devices",
        type=int,
        default=default_n_devices,
        help="number of devices (default: {:d})".format(default_n_devices),
    )
    parser.add_argument(
        "--partition",
        choices=partition_choices,
        default=default_partition,
        help="['dirichlet', 'homo', 'byclass', 'dirichlet_uniform']",
    )
    parser.add_argument(
        "--dataset",
        choices=dataset_choices,
        default="mnist",
        help="['mnist']",
    )
    parser.add_argument("--neighbor", action="store_true")
    parser.add_argument(
        "--alpha",
        type=float,
        default=default_alpha,
        help=f"dirichlet alpha. Default: {default_alpha}",
    )
    parser.add_argument(
        "--minsize",
        type=int,
        default=20,
        help=f"dirichlet alpha. Default: {default_alpha}",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=default_ratio,
        help="ratio of total devices to replace with uniform",
    )
    parser.add_argument(
        "--n_classes_per_device",
        type=int,
        default=default_n_classes_per_device,
        help=f"number of classes per device. Default: {default_n_classes_per_device}",
    )
    parser.add_argument("--seed", type=int, default=default_seed, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    if args.dataset == "mnist":
        dataset = "mnist"
        data_dir = "data/mnist"
    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        test_dataidx_map,
        path_str,
    ) = partition_data(dataset, data_dir, args)

    with open(
        f"data/{dataset}_split/{path_str}.json", "w"
    ) as fp:
        json.dump(net_dataidx_map, fp)

    if test_dataidx_map != None:
        with open(
            f"data/{dataset}_split/{path_str}_test.json", "w"
        ) as fp:
            json.dump(test_dataidx_map, fp)
