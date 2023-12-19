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


class Finetune(AGEM):
    # Implementation is partially based on: https://github.com/MehdiAbbanaBennani/continual-learning-ogdplus
    def __init__(self, backbone, benchmark, params):
        self.backbone = backbone
        self.benchmark = benchmark
        self.params = params
        
        super(Finetune, self).__init__(backbone, benchmark, params)
    
    def sample_batch_from_memory_sensitive(self):
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
            inp_ref, targ_ref, task_ids_ref = self.sample_batch_from_memory()
            pred_ref = self.backbone(inp_ref, task_ids_ref)
            loss = criterion(pred_ref, targ_ref.reshape(len(targ_ref)))
            loss.backward()
            grad_ref = flatten_grads(self.backbone).detach().clone()
            grad_batch += 0.1*grad_ref
            optimizer.zero_grad()
            self.backbone = assign_grads(self.backbone, grad_batch)
        optimizer.step()
