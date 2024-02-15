import data
import copy
import util
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import json
import torch.utils.data as data

from device_race import Device
from race_srp import SRP_Gaussin_torch, get_race_distance

SPLITPATH_TRAIN = {
    "mnist_dirichlet_uniform":""
}

DATAPATH = {
    "mnist": "../data/mnist",
}
CONFIG = {
    "mnist": {"d": 784, "c": 10},
}

def fed_race_personalize(args):
    print("Running Personalization")
    args_str = f"Round{args.T}_Epoch{args.E}_Batch{args.batch_size}_LR{args.learn_rate}_Device{args.K}"
    print(args_str)

    save_path = (
        "../artifact/racefl/" + args.dataset + "/personalize_global_model/" + args_str
    )
    if args.tensorboard:
        tb_path = "../artifact/racefl/" + args.dataset + "/personalize_" + args_str
        print("tensorboard path", tb_path)
        writer = SummaryWriter(tb_path)

    # read data file
    with open(SPLITPATH_TRAIN[args.dataset], "r") as fp:
        train_net_dataidx_map = json.load(fp)

    test_net_dataidx_map = {}
    for id in train_net_dataidx_map.keys():
        test_net_dataidx_map[id] = None

    if args.dataset.startswith("mnist"):
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_ds = datasets.MNIST(
            DATAPATH["mnist"], train=True, transform=transform, download=True
        )
        test_ds = datasets.MNIST(
            DATAPATH["mnist"], train=False, transform=transform, download=True
        )
        IN = CONFIG["mnist"]["d"]
        OUT = CONFIG["mnist"]["c"]
        test_loader = data.DataLoader(
            dataset=test_ds, batch_size=args.batch_size, shuffle=False
        )

    print("data dimension =", IN)
    print("# classes =", OUT)
    print("# of test = ", len(test_ds))
    print("# of train = ", len(train_ds))

    # initialize devices
    devices = []
    for id in train_net_dataidx_map.keys():
        devices.append(
            Device(
                id,
                train_net_dataidx_map[id],
                test_net_dataidx_map[id],
                train_ds,
                test_ds,
                IN,
                OUT,
                args,
            )
        )

    cold_starter = devices[: args.N]
    devices = devices[args.N :]

    print("# training devices =", len(devices))
    print("# cold start devices =", len(cold_starter))

    if args.train_global:
        model_params = devices[0].get_model()
        param_buffer = []
        total_param = 0
        for p in model_params.parameters():
            if p.requires_grad:
                total_param += 1
                param_buffer.append(p.data.detach().clone())
        param_avg = copy.deepcopy(param_buffer)
        p_i = 0
        for p in param_buffer:
            param_buffer[p_i] = torch.zeros_like(param_buffer[p_i], requires_grad=False)
            p_i += 1

        best_testacc = 0
        wait_round = 0
        best_param = copy.deepcopy(param_buffer)
        for t in range(args.T):
            devices_sample = np.random.choice(devices, args.K, replace=False)
            # set up device for training
            for dev_i in devices_sample:
                dev_i.setup_for_training()

            local_norm_weights = util.get_norm_weights_devices(devices_sample)
            devices_train_acc = []
            devices_test_acc = []
            for dev_i in devices_sample:

                dev_i.set_weights(param_avg)
                param_result, train_acc = dev_i.train()

                devices_train_acc += [
                    train_acc
                ]  # local model's accuracy on valid ation set

                p_i = 0
                for param in param_result:
                    param_buffer[p_i] += param * local_norm_weights[dev_i.get_id()]
                    p_i += 1

            param_avg = copy.deepcopy(param_buffer)
            # reinitialize parameter buffer
            p_i = 0
            for p in param_buffer:
                param_buffer[p_i] = torch.zeros_like(
                    param_buffer[p_i], requires_grad=False
                )
                p_i += 1

            # test each sampled device
            for dev_i in devices_sample:
                dev_i.set_weights(param_avg)
                devices_test_acc += [dev_i.evaluate()]

            # test global model on test set
            test_accuracy = devices_sample[0].evaluate(dataloader=test_loader)
            print(
                f"[{t+1} / {args.T} ] Train acc: {np.mean(devices_train_acc):.4f}, Local test acc : {np.mean(devices_test_acc):.4f}, Global Test acc: {test_accuracy:.4f}"
            )

            if args.tensorboard:
                writer.add_scalar("Acc/train", np.mean(devices_train_acc), t)
                writer.add_scalar("Acc/local_test", np.mean(devices_test_acc), t)
                writer.add_scalar("Acc/test", test_accuracy, t)

            if test_accuracy >= best_testacc:
                best_testacc = test_accuracy
                best_param = param_avg
                wait_round = 0
            else:
                wait_round += 1

            if wait_round >= 30:
                break

        print("History Test Accuracy {:.4f}".format(best_testacc))
        torch.save(best_param, save_path + ".pt")
    else:
        print(f"Loading Global model from {save_path}.pt")
        best_param = torch.load(save_path + ".pt")

    print("===Global Model ===")
    global_model_device_test_acc = []
    for dev_i in devices:
        dev_i.setup_for_training()
        dev_i.set_weights(best_param)
        global_model_device_test_acc += [dev_i.evaluate()]
    print(
        f" Global Model test on warm device: {np.array(global_model_device_test_acc).mean():.4f}"
    )
    global_model_device_test_acc = []
    for dev_i in cold_starter:
        dev_i.setup_for_training()
        dev_i.set_weights(best_param)
        global_model_device_test_acc += [dev_i.evaluate()]
    print(
        f" Global Model test on cold device: {np.array(global_model_device_test_acc).mean():.4f}"
    )
    print(global_model_device_test_acc)

    print("===Local Model===")
    print("Finetune on Local Device")
    save_path = (
        "../artifact/racefl/" + args.dataset + "/personalize_local_model/" + args_str
    )

    if args.train_local:
        device_param_dic = {}
        for dev_i in devices:
            dev_i.setup_for_training()
            dev_i.set_weights(best_param)
            dev_i_param, train_acc = dev_i.train()
            device_param_dic[dev_i.get_id()] = copy.deepcopy(dev_i_param)
        torch.save(device_param_dic, save_path + ".pt")
    else:
        print(f"Loading local model from {save_path}.pt")
        device_param_dic = torch.load(save_path + ".pt")

    local_model_device_test_acc = []
    for dev_i in devices:
        dev_i.set_weights(device_param_dic[dev_i.get_id()])
        local_model_device_test_acc += [dev_i.evaluate()]
    print(
        f"Personalized Model Accuracy {np.array(local_model_device_test_acc).mean(): .4f} \n {local_model_device_test_acc}",
    )

    hashes = SRP_Gaussin_torch(args.raceK, args.raceR, IN, args.seed)

    sketch_dict = {}
    sketch_raw = {}
    for device in devices:
        device_s, device_n = device.sketch_input(args.raceK, args.raceR, IN, hashes)
        sketch_dict[device.get_id()] = device_s / device_n
        sketch_raw[device.get_id()] = device_s
        sketch_raw[device.get_id()] = device_n


    local_cold_starter_test_acc = []
    model_params = devices[0].get_model()
    for dev_i in cold_starter:
        dev_i.setup_for_training()
        # retrieve personalize model

        accuracy_buffer = []
        for d in devices:
            neighbors = [d.get_id()]

            param_buffer = []

            for p in model_params.parameters():
                if p.requires_grad:
                    param_buffer.append(p.data.detach().cpu().clone())
            p_i = 0
            for p in param_buffer:
                param_buffer[p_i] = torch.zeros_like(
                    param_buffer[p_i], requires_grad=False
                )
                p_i += 1

            for n in neighbors:
                param_result = device_param_dic[n]

                p_i = 0
                for param in param_result:
                    param_buffer[p_i] += param.cpu()
                    p_i += 1

            p_i = 0
            for param in param_result:
                param_buffer[p_i] /= len(neighbors)
                p_i += 1

            neighr_acc = dev_i.set_weights(param_buffer)
            neighr_acc = dev_i.evaluate()
            accuracy_buffer += [neighr_acc]

        local_cold_starter_test_acc += [max(accuracy_buffer)]

    print("=====Cold Starter =====")
    print(
        f"Nearest Neighbor Model Accuracy {np.array(local_cold_starter_test_acc).mean() : .4f} \n {local_cold_starter_test_acc}",
    )
