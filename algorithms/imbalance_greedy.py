import torch
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.agem import AGEM
from cl_gym.algorithms.utils import flatten_grads, assign_grads
from torch.nn.functional import relu, avg_pool2d
import matplotlib.pyplot as plt

from algorithms.imbalance import Heuristic2


import copy
import os

def bool2idx(arr):
    idx = list()
    for i, e in enumerate(arr):
        if e == 1:
            idx.append(i)
    return np.array(idx)


class Heuristic1(Heuristic2):
    def prepare_train_loader(self, task_id, epoch=0):
        """
        Compute gradient for memory replay
        Compute individual sample gradient (against buffer data) for all current data
            loader로 불러와서 모든 output과 embedding을 저장
            gradient 계산 (W, b)
        Return train loader
        """
        num_workers = self.params.get('num_dataloader_workers', torch.get_num_threads())
        if task_id == 1: # no memory
            return self.benchmark.load(task_id, self.params['batch_size_train'],
                                    num_workers=num_workers, pin_memory=True)[0]
        
        if self.params['alpha'] == 0:
            return self.benchmark.load(task_id, self.params['batch_size_train'],
                                    num_workers=num_workers, pin_memory=True)[0]
        
        if epoch <= 1:
            self.original_seq_indices_train = self.benchmark.seq_indices_train[task_id]
            if hasattr(self.benchmark.trains[task_id], "sensitive"):
                print(f"Num. of sensitives: {(self.benchmark.trains[task_id].sensitive[self.original_seq_indices_train] != self.benchmark.trains[task_id].targets[self.original_seq_indices_train]).sum().item()}")
        else:
            self.benchmark.seq_indices_train[task_id] = copy.deepcopy(self.original_seq_indices_train)
        self.non_select_indexes = list(range(len(self.benchmark.seq_indices_train[task_id])))
        
        losses, n_grads_all, n_r_new_grads = self.get_loss_grad_all(task_id)
        loss_matrix = losses.repeat(len(n_r_new_grads), 1)
        forget_matrix = torch.matmul(n_r_new_grads, torch.transpose(n_grads_all, 0, 1))


        # current data selection
        accumulate_select_indexes = []
        accumulate_sum = []
        select_indexes = []

        # for debug
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=False,
                                   num_workers=num_workers, pin_memory=True)[0]
        targets = train_loader.dataset.targets \
            if hasattr(train_loader.dataset, "targets") else train_loader.dataset.dataset.targets
        if isinstance(targets, torch.Tensor):
            classes = targets.unique().cpu().numpy()
        elif isinstance(targets, np.ndarray):
            classes = np.unique(targets)
        num_dict = {x:0 for x in classes}
        num_dict_list = {x:list() for x in classes}

        # data_len = len(targets)
        data_len = len(loss_matrix)

        # for debugging
        classwise_loss = []


        for b in range(data_len-1):
            loss_matrix = loss_matrix - self.params['alpha'] * forget_matrix
            loss_mean = torch.mean(loss_matrix, dim=1, keepdim=True)
            loss_std = torch.std(loss_matrix, dim=1, keepdim=True)

            select_ind = torch.argmin(loss_mean + loss_std, dim=0)
            accumulate_sum.append(copy.deepcopy(loss_mean[select_ind].item() + loss_std[select_ind].item()))
            classwise_loss.append(loss_matrix[select_ind].view(-1).detach().clone().cpu().numpy())

            target_idx = self.non_select_indexes[select_ind.item()]
            # print(f"{targets[target_idx].item()=}")
            num_dict[targets[target_idx].item()] += 1
            for k in num_dict:
                num_dict_list[k].append(num_dict[k])

            select_indexes.append(self.non_select_indexes[select_ind.item()])
            accumulate_select_indexes.append(copy.deepcopy(select_indexes))
            
            # del self.benchmark.seq_indices_train[task_id][select_ind.item()]
            del self.non_select_indexes[select_ind.item()]

            best_buffer_losses = loss_matrix[select_ind].view(1,-1)
            loss_matrix = best_buffer_losses.repeat(len(loss_matrix)-1, 1)

            n_r_new_grads = torch.cat((n_r_new_grads[:select_ind.item()], n_r_new_grads[select_ind.item()+1:]))
            forget_matrix = torch.cat((forget_matrix[:select_ind.item()], forget_matrix[select_ind.item()+1:]))

        # for debugging
        os.makedirs(f"{self.params['output_dir']}/figs", exist_ok=True)
        plt.plot(accumulate_sum)
        plt.savefig(f"{self.params['output_dir']}/figs/tid_{task_id}_epoch_{epoch}_accumulate_loss.png")
        plt.clf()

        classwise_loss = np.array(classwise_loss).T
        for i, e in enumerate(classwise_loss):
            plt.plot(e, label=i)
        plt.legend(loc="best")
        plt.savefig(f"{self.params['output_dir']}/figs/tid_{task_id}_epoch_{epoch}_classwise_loss.png")

        plt.clf()

        for i, e in num_dict_list.items():
            plt.plot(e, label=i)
        plt.legend(loc="best")
        plt.savefig(f"{self.params['output_dir']}/figs/tid_{task_id}_epoch_{epoch}_class_num.png")

        plt.clf()

        best_ind = np.argmin(np.array(accumulate_sum))
        select_curr_indexes = accumulate_select_indexes[best_ind]

        # select_curr_indexes = np.random.choice(len(self.benchmark.seq_indices_train[task_id]), 1000, False)
        print(f"{len(select_curr_indexes)=}")

        select_curr_indexes = list(set(select_curr_indexes))
        select_curr_indexes.sort()
        print(f"For debugging: {len(select_curr_indexes)=}")

        self.benchmark.seq_indices_train[task_id] = select_curr_indexes
        
        num_workers = self.params.get('num_dataloader_workers', torch.get_num_threads())
        return self.benchmark.load(task_id, self.params['batch_size_train'],
                                   num_workers=num_workers, pin_memory=True)[0]