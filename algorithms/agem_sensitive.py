import torch
import numpy as np
from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.agem import AGEM
from cl_gym.algorithms.utils import flatten_grads, assign_grads

def bool2idx(arr):
    idx = list()
    for i, e in enumerate(arr):
        if e == 1:
            idx.append(i)
    return np.array(idx)


class AGEM_Sensitive(AGEM):
    # Implementation is partially based on: https://github.com/MehdiAbbanaBennani/continual-learning-ogdplus
    def __init__(self, backbone, benchmark, params):
        self.backbone = backbone
        self.benchmark = benchmark
        self.params = params
        
        super(AGEM_Sensitive, self).__init__(backbone, benchmark, params)
    
    @staticmethod
    def __is_violating_direction_constraint(grad_ref, grad_batch):
        """
        GEM and A-GEM operate on gradient directions.
        i.e., gradient direction should have angle less than 90 degrees with reference gradient.
        :param grad_ref: reference gradient (i.e., grads on episodic memory)
        :param grad_batch: batch gradient
        :return:
        """
        return torch.dot(grad_ref, grad_batch) < 0
    
    @staticmethod
    def __project_grad_vector(grad_ref, grad_batch):
        """
        A-GEM operates on regularized average gradient directions.
        In case of violation, gradients should be projected (see Eq.(11) in A-GEM paper).
        :param grad_ref: reference gradient (i.e., grads on episodic memory examples)
        :param grad_batch: current batch gradients
        :return: projected gradients
        """
        return grad_batch - (torch.dot(grad_batch, grad_ref) / torch.dot(grad_ref, grad_ref)) * grad_ref

    def sample_batch_from_memory(self):
        try:
            batch = next(self.episodic_memory_iter)
        except StopIteration:
            self.episodic_memory_iter = iter(self.episodic_memory_loader)
            batch = next(self.episodic_memory_iter)
        
        device = self.params['device']
        inp, targ, task_id, sensitive = batch
        return inp.to(device), targ.to(device), task_id.to(device), sensitive


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

            #get grad_ref_s (grad_ref of sensitive group)
            inp_ref_s, targ_ref_s = inp_ref[sensitive_ref], targ_ref[sensitive_ref]
            pred_ref_s = self.backbone(inp_ref_s, task_ids_ref)
            loss = criterion(pred_ref_s, targ_ref_s.reshape(len(targ_ref_s)))
            loss.backward()
            grad_ref_s = flatten_grads(self.backbone).detach().clone()

            # if self.__is_violating_direction_constraint(grad_ref, grad_batch):
                # print(f"{self.__is_violating_direction_constraint(grad_ref, grad_batch)=}")
                # print(f"{torch.dot(grad_ref, grad_batch)=}")
                # print(f"{torch.linalg.norm(grad_batch)}")
                # grad_ref = self.__project_grad_vector(grad_ref, grad_batch)
                # grad_ref = self.__project_grad_vector(grad_batch, grad_ref)
                # grad_batch = self.__project_grad_vector(grad_ref, grad_batch)
                # print(f"MODIFY GRAD_BATCH")
                # print(f"{torch.dot(grad_ref, grad_batch)=}")
                # print(f"{torch.linalg.norm(grad_batch)}")
                # print()

            grad_batch += 0.1*grad_ref

            optimizer.zero_grad()

            # self.backbone = assign_grads(self.backbone, grad_ref)
            self.backbone = assign_grads(self.backbone, grad_batch)
        optimizer.step()
