import torch
import torch.nn as nn
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.test_utils import *
from utils.data_utils import *
from utils.crypto_utils import *
from utils.globals import *
from utils.attack_utils import *
import copy

class ClientUpdate(object):
    def __init__(self, dataset, batchSize, learning_rate, epochs, idxs, k, args):
        self.train_loader = DataLoader(CustomDataset(dataset, idxs), batch_size=batchSize, shuffle=True)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.k = k # index of the client
        self.adv_thd = int(args.attack_prop * args.client_number)
        self.num_classes = data2label[args.dataset]
        self.args = args

    def train(self, model):
        
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.95, weight_decay = 5e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        old_weights = copy.deepcopy(model.state_dict())
        e_loss = []
        for epoch in range(1, self.epochs + 1):

            train_loss = 0.0

            model.train()
            for data, labels in self.train_loader:
                if data.size()[0] < 2:
                    continue

                data, labels = data.to(self.args.device), labels.to(self.args.device)
                if self.args.attack_type == "label_flip":
                    if self.args.defense == "rflpa":
                        if self.k <= self.adv_thd and self.k > 0:
                            labels = self.num_classes - labels - 1
                    else:
                        if self.k < self.adv_thd:
                            labels = self.num_classes - labels - 1
                elif self.args.attack_type in bd_attacks:
                    if self.args.defense == "rflpa":
                        if self.k <= self.adv_thd and self.k > 0:
                            data = add_trigger(data, add_all = True, args=self.args)
                            labels = self.num_classes - labels - 1
                        # elif self.k % 5 == 0:
                        #     data = add_trigger(data, add_all = False, args=self.args)
                    else:
                        if self.k < self.adv_thd:
                            data = add_trigger(data, add_all = True, args=self.args)
                            labels = self.num_classes - labels - 1
                        # elif self.k % 5 == 0:
                        #     data = add_trigger(data, add_all = False, args=self.args)

                # if self.k < self.adv_thd:
                #     print(data[0])
                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = model(data)
                # calculate the loss
                loss = criterion(output, labels)
                scaling_attack = False
                if self.args.attack_type == "scaling":
                    if self.args.defense == "rflpa":
                        if self.k <= self.adv_thd and self.k > 0:
                            scaling_attack = True
                    else:
                        if self.k < self.adv_thd:
                            scaling_attack = True
                if scaling_attack:
                    distance_loss = model_dist_norm_var(model, old_weights)
                    # print(distance_loss)
                    loss = 0.8 * loss + 0.2 * distance_loss
                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)
            # average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)

            # self.learning_rate = optimizer.param_groups[0]['lr']

        total_loss = sum(e_loss) / len(e_loss)
        new_weights = copy.deepcopy(model.state_dict())
        deltas = {}
        for key in old_weights:
            this_delta = new_weights[key] - old_weights[key]
            if self.args.defense in ["rflpa", "brea"]:
                this_delta = torch.clip(this_delta, -self.args.clip, self.args.clip)
                this_delta = (this_delta * self.args.precision).long()
                this_delta = this_delta / self.args.precision
                new_weights[key] = old_weights[key] + this_delta
            deltas[key] = this_delta
        return deltas, new_weights, total_loss
    
    def train_dp(self, model, mode="central"):
        criterion = nn.CrossEntropyLoss(reduction='none')
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.95, weight_decay = 5e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        old_weights = copy.deepcopy(model.state_dict())
        e_loss = []
        for epoch in range(1, self.epochs + 1):

            train_loss = 0.0

            model.train()
            for data, labels in self.train_loader:
                clipped_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
                if data.size()[0] < 2:
                    continue

                data, labels = data.to(self.args.device), labels.to(self.args.device)
                if self.args.attack_type == "label_flip":
                    if self.args.defense == "rflpa":
                        if self.k <= self.adv_thd and self.k > 0:
                            labels = self.num_classes - labels - 1
                    else:
                        if self.k < self.adv_thd:
                            labels = self.num_classes - labels - 1
                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = model(data)
                # calculate the loss
                loss = criterion(output, labels)
                for i in range(loss.size()[0]):
                    train_loss += loss[i].item()
                    loss[i].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.norm_clip)
                    for name, param in model.named_parameters():
                        clipped_grads[name] += param.grad 
                    model.zero_grad()

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        this_grad = clipped_grads[name]
                        if mode=="local":
                            this_grad += (gaussian_noise(this_grad.shape, self.args) / len(self.train_loader))
                        param.grad = (this_grad/loss.size()[0])

                # perform a single optimization step
                optimizer.step()
            # average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)

            # self.learning_rate = optimizer.param_groups[0]['lr']

        total_loss = sum(e_loss) / len(e_loss)
        new_weights = copy.deepcopy(model.state_dict())

        deltas = {}
        for key in old_weights:
            this_delta = new_weights[key] - old_weights[key]
            deltas[key] = this_delta
        return deltas, new_weights, total_loss