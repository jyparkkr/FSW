import torch
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.agem import AGEM
from cl_gym.algorithms.utils import flatten_grads, assign_grads
from torch.nn.functional import relu, avg_pool2d
from algorithms.mean_std_min import Heuristic2

import matplotlib.pyplot as plt

import copy
import os

def bool2idx(arr):
    idx = list()
    for i, e in enumerate(arr):
        if e == 1:
            idx.append(i)
    return np.array(idx)


class Heuristic3(Heuristic2):
    # Implementation is partially based on: https://github.com/MehdiAbbanaBennani/continual-learning-ogdplus
    def __init__(self, backbone, benchmark, params, **kwargs):
        self.backbone = backbone
        self.benchmark = benchmark
        self.params = params
        
        super(Heuristic2, self).__init__(backbone, benchmark, params, **kwargs)

    def memory_indices_selection(self, task):
        ## update self.benchmark.memory_indices_train[task] with len self.benchmark.per_task_memory_examples
        
        indices_train = np.arange(self.per_task_memory_examples)
        # num_examples = self.benchmark.per_task_memory_examples
        # indices_train = self.sample_uniform_class_indices(self.trains[task], start_cls, end_cls, num_examples)
        # # indices_test = self.sample_uniform_class_indices(self.tests[task], start_cls, end_cls, num_examples)
        assert len(indices_train) == self.per_task_memory_examples
        self.benchmark.memory_indices_train[task] = indices_train[:]

    def update_episodic_memory(self):
        # self.memory_indices_selection(self.current_task)
        self.episodic_memory_loader, _ = self.benchmark.load_memory_joint(self.current_task,
                                                                          batch_size=self.params['batch_size_memory'],
                                                                          shuffle=True,
                                                                          pin_memory=True)
        self.episodic_memory_iter = iter(self.episodic_memory_loader)


    def sample_batch_from_memory(self):
        try:
            batch = next(self.episodic_memory_iter)
        except StopIteration:
            self.episodic_memory_iter = iter(self.episodic_memory_loader)
            batch = next(self.episodic_memory_iter)
        
        device = self.params['device']
        inp, targ, task_id, *_ = batch
        return inp.to(device), targ.to(device), task_id.to(device), _

    def training_task_end(self):
        """
        Select what to store in the memory in this step
        """
        print("training_task_end")
        if self.requires_memory:
            self.update_episodic_memory()
        self.current_task += 1

    def forward_embeds(self, inp):
        if self.params['dataset'] in ['MNIST', "FMNIST"]: #MLP
            inp = inp.view(inp.shape[0], -1)
            out = inp
            for block in self.backbone.blocks:
                embeds = out
                out = block(out)
            return out, embeds
        else: #ResNet
            bsz = inp.size(0)
            shape = (bsz, self.backbone.dim, \
                        self.backbone.input_shape[-2], self.backbone.input_shape[-1])
            out = relu(self.backbone.bn1(self.backbone.conv1(inp.view(shape))))
            out = self.backbone.layer1(out)
            out = self.backbone.layer2(out)
            out = self.backbone.layer3(out)
            out = self.backbone.layer4(out)
            out = avg_pool2d(out, 4)
            embeds = out.view(out.size(0), -1)
            out = self.backbone.linear(embeds)
            return out, embeds

    def get_loss_grad(self, task_id, loader, current_set = False):
        criterion = self.prepare_criterion(task_id)
        device = self.params['device']
        sensitive_loss = {x:list() for x in range(2)}
        sensitive_grad_dict = {x:list() for x in range(2)}
        new_grads = None

        for batch_idx, (inp, targ, t_id, *_) in enumerate(loader):
            # self.backbone.forward
            inp, targ, t_id  = inp.to(device), targ.to(device), t_id.to(device)
            sen = _[0].to(device) # ADDED
            pred, embeds = self.forward_embeds(inp)
            criterion.reduction = "none"
            loss = criterion(pred, targ.reshape(-1))
            criterion.reduction = "mean"
            
            bias_grads = torch.autograd.grad(loss.mean(), pred)[0]
            bias_expand = torch.repeat_interleave(bias_grads, embeds.shape[1], dim=1)
            weight_grads = bias_expand * embeds.repeat(1, pred.shape[1])
            grads = torch.cat([bias_grads, weight_grads], dim=1)

            for i, e in enumerate(sen):
                sensitive_loss[e.cpu().item()].append(loss[i])
                sensitive_grad_dict[e.cpu().item()].append(grads[i])

            if current_set:
                new_grads = grads if new_grads is None else torch.cat([new_grads, grads])

            self.backbone.zero_grad()

        for x in range(2):
            if len(sensitive_loss[x]) == 0:
                del sensitive_loss[x]
                del sensitive_grad_dict[x]
        
        return sensitive_loss, sensitive_grad_dict, new_grads
    
    def get_loss_grad_all(self, task_id):
        num_workers = self.params.get('num_dataloader_workers', 0)

        sensitive_loss, sensitive_grad_dict, _ = self.get_loss_grad(task_id, self.episodic_memory_loader, current_set = False)
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=False,
                                   num_workers=num_workers, pin_memory=True)[0]
        current_loss, current_grad_dict, new_grads = self.get_loss_grad(task_id, train_loader, current_set = True)
        sensitive_loss.update(current_loss)
        sensitive_grad_dict.update(current_grad_dict)

        losses = []
        grads = []
        for k, v in sensitive_loss.items():
            v3 = sensitive_grad_dict[k]
            losses.append(torch.stack(v).mean(dim=0).view(1, -1).detach().clone())
            grads.append(torch.stack(v3).mean(dim=0).view(1, -1).detach().clone())

        with torch.no_grad():
            losses = torch.cat(losses, dim=0).view(1,-1)
            buffer_grads = torch.cat(grads, dim=0)

            n_buffer_grads = F.normalize(buffer_grads, p=2, dim=1)
            n_new_grads = F.normalize(new_grads, p=2, dim=1)

            loss_matrix = losses.repeat(len(n_new_grads), 1)
            forget_matrix = torch.matmul(n_new_grads, torch.transpose(n_buffer_grads, 0, 1))

        return loss_matrix, forget_matrix, n_new_grads



    def prepare_train_loader(self, task_id):
        """
        Compute gradient for memory replay
        Compute individual sample gradient (against buffer data) for all current data
            loader로 불러와서 모든 output과 embedding을 저장
            gradient 계산 (W, b)
        각 batch별 loss와 std를 가장 낮게 하는 (하나)의 sample만 취해서 학습에 사용
        Return train loader
        """
        num_workers = self.params.get('num_dataloader_workers', 0)
        if task_id == 1: # no memory
            return self.benchmark.load(task_id, self.params['batch_size_train'],
                                    num_workers=num_workers, pin_memory=True)[0]
        
        loss_matrix, forget_matrix, n_new_grads = self.get_loss_grad_all(task_id)
        print(f"{loss_matrix.shape=}")

        # current data selection
        accumulate_select_indexes = []
        accumulate_sum = []
        select_indexes = []

        # targets = train_loader.dataset.targets \
        #     if hasattr(train_loader.dataset, "targets") else train_loader.dataset.dataset.targets
        # data_len = len(targets)
        data_len = len(loss_matrix)
        non_select_indexes = list(range(data_len))

        # for debugging
        classwise_loss = []

        for b in range(data_len-1):
            # inp, targ, t_id  = inp.to(device), targ.to(device), t_id.to(device)
            loss_matrix = loss_matrix - self.params['alpha'] * forget_matrix
            loss_mean = torch.mean(loss_matrix, dim=1, keepdim=True)
            loss_std = torch.std(loss_matrix, dim=1, keepdim=True)

            loss_1 = loss_matrix[:,1]

            # select_ind = torch.argmin(loss_mean + loss_std, dim=0)
            select_ind = torch.argmin(loss_1, dim=0)
            accumulate_sum.append(copy.deepcopy(loss_1[select_ind].item()))
            # accumulate_sum.append(copy.deepcopy(loss_mean[select_ind].item() + loss_std[select_ind].item()))



            classwise_loss.append(loss_matrix[select_ind].view(-1).detach().clone().cpu().numpy())

            # num_dict[targets[select_ind.item()].item()] += 1

            select_indexes.append(non_select_indexes[select_ind.item()])
            accumulate_select_indexes.append(copy.deepcopy(select_indexes))
            
            del self.benchmark.seq_indices_train[task_id][select_ind.item()]
            del non_select_indexes[select_ind.item()]

            # loss_matrix, forget_matrix = self.get_loss_grad_all(task_id)

            best_buffer_losses = loss_matrix[select_ind].view(1,-1)
            loss_matrix = best_buffer_losses.repeat(len(n_new_grads)-1, 1)

            n_new_grads = torch.cat((n_new_grads[:select_ind.item()], n_new_grads[select_ind.item()+1:]))
            forget_matrix = torch.cat((forget_matrix[:select_ind.item()], forget_matrix[select_ind.item()+1:]))

        # for debugging
        save_path = f"./figs/sensitive_2/alpha_{self.params['alpha']}"
        os.makedirs(save_path, exist_ok=True)
        plt.plot(accumulate_sum)
        plt.savefig(f"{save_path}/tid_{task_id}_accumulate_loss.png")
        plt.clf()

        classwise_loss = np.array(classwise_loss).T
        for i, e in enumerate(classwise_loss):
            plt.plot(e, label=i)
        plt.legend(loc="best")
        plt.savefig(f"{save_path}/tid_{task_id}_classwise_loss.png")
        plt.clf()


        best_ind = np.argmin(np.array(accumulate_sum))
        select_curr_indexes = accumulate_select_indexes[best_ind]
        print(f"{len(select_curr_indexes)=}")

        select_curr_indexes = list(set(select_curr_indexes))
        self.benchmark.seq_indices_train[task_id] = select_curr_indexes
        
        num_workers = self.params.get('num_dataloader_workers', 0)
        return self.benchmark.load(task_id, self.params['batch_size_train'],
                                   num_workers=num_workers, pin_memory=True)[0]


    def training_step(self, task_ids, inp, targ, optimizer, criterion, sensitive=None):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_ids)
        loss = criterion(pred, targ)
        loss.backward()
        if task_ids[0] > 1:
            grad_batch = flatten_grads(self.backbone).detach().clone()
            optimizer.zero_grad()

            # get grad_ref
            inp_ref, targ_ref, task_ids_ref, sensitive_ref = self.sample_batch_from_memory()
            pred_ref = self.backbone(inp_ref, task_ids_ref)
            loss = criterion(pred_ref, targ_ref.reshape(len(targ_ref)))
            loss.backward()
            grad_ref = flatten_grads(self.backbone).detach().clone()
            grad_batch += self.params['lambda']*grad_ref

            optimizer.zero_grad()
            self.backbone = assign_grads(self.backbone, grad_batch)
        optimizer.step()
