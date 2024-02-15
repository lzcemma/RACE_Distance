import numpy as np
import torch
import model
from race_srp import RACE_SRP
from datasets import MNIST_truncated
import torch.utils.data as data
import copy
import random


class Device:
    def __init__(self, id, train_idx, test_idx, train_ds, test_ds, IN, OUT, args):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
        self.pdevice = torch.device(dev)

        self.id = id
        if args.dataset.startswith("mnist"):
            self.model = model.MLP([IN, 512], OUT)
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

        self.lr = args.learn_rate
        self.n_epochs = args.E
        self.batch_size = args.batch_size

        self.all_data = train_ds
        self.localdata_idx = train_idx
        self.test_data = test_ds
        self.localtest_idx = test_idx
        self.dataset = args.dataset

        self.train_ds = None
        self.valid_ds = None
        self.train_loader = None
        self.valid_loader = None

    def get_id(self):
        return self.id

    def get_model(self):
        return self.model

    def set_weights(self, param_buffer):
        p_i = 0
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = param_buffer[p_i].detach().clone()
                p_i += 1

    def get_ntrain(self):
        return len(self.localdata_idx)

    def get_weights(self):
        return self.model.get_weights()

    def sketch_input(self, K, R, d, hashes):
        race = RACE_SRP(R, K, d, hashes)
        alpha = np.ones(len(self.localdata_idx))

        # need data loader to apply same transform
        if self.dataset.startswith("mnist"):
            if self.train_ds is None:
                self.train_ds = MNIST_truncated(self.all_data, self.localdata_idx)
        else:
            raise NotImplementedError
        temp_loader = data.DataLoader(
            dataset=self.train_ds, batch_size=len(self.localdata_idx), shuffle=False
        )
        for x, _ in temp_loader:
            if x.dim() > 2:
                x = x.reshape(len(x), -1)

            race.add(x, alpha, self.pdevice)
            break

        return race.counts, race.N

    def setup_for_training(self):
        num_train = int(0.9 * (len(self.localdata_idx)))
        random.shuffle(self.localdata_idx)

        train_idx = self.localdata_idx[:num_train]
        valid_idx = self.localdata_idx[-num_train:]
        if self.dataset.startswith("mnist"):
            self.train_ds = MNIST_truncated(self.all_data, train_idx)
            self.valid_ds = MNIST_truncated(self.all_data, valid_idx)
            self.test_ds = MNIST_truncated(self.test_data, self.localtest_idx)
        
        self.train_loader = data.DataLoader(
            dataset=self.train_ds, batch_size=self.batch_size, shuffle=True
        )
        self.valid_loader = data.DataLoader(
            dataset=self.valid_ds, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = data.DataLoader(
            dataset=self.test_ds, batch_size=self.batch_size, shuffle=False
        )

    def train(self, args=None):
        # print(np.unique(np.array(self.train_ds.target)))
        # print(len((self.train_ds)))
        if args is not None:
            self.lr = args.learn_rate
            self.n_epochs = args.E
        self.model.train()
        self.model.to(self.pdevice)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        best_acc = 0.0
        best_model = None
        wait = 0
        for _ in range(self.n_epochs):
            for batch_idx, (x, target) in enumerate(self.train_loader):
                self.model.zero_grad()

                if self.dataset.startswith("mnist"):
                    x = x.reshape(len(target), -1)
                
                y_hat = self.model(x.to(self.pdevice))

                loss = self.loss_fn(y_hat, target.to(self.pdevice))
                loss.backward()
                self.optimizer.step()

            train_acc = self.evaluate(self.valid_loader)
            if train_acc >= best_acc:
                best_acc = train_acc
                best_model = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
            if wait >= 5:
                break
        self.model.load_state_dict(best_model)
        self.model.cpu()
        param_buffer = []
        for p in self.model.parameters():
            if p.requires_grad:
                param_buffer.append(p.data.cpu().detach().clone())

        return param_buffer, train_acc

    def evaluate(self, dataloader=None, return_pred=False):
        if dataloader is None:
            dataloader = self.test_loader

        self.model.eval()
        self.model.to(self.pdevice)
        true_labels_list, pred_labels_list = np.array([]), np.array([])

        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                if self.dataset.startswith("mnist"):
                    x = x.reshape(len(target), -1)

                out = self.model(x.to(self.pdevice)).cpu()
                _, pred_label = torch.max(out.data, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
        if return_pred:
            return correct / float(total), (pred_labels_list, true_labels_list)
        else:
            return correct / float(total)
