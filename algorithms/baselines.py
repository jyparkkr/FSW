import torch
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F
from typing import Tuple, Optional, Dict, List
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset

from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.utils import flatten_grads, assign_grads
import time
import importlib
import copy
import os

class BaseContinualAlgoritm(ContinualAlgorithm):
    def sample_batch_from_memory(self):
        try:
            batch = next(self.episodic_memory_iter)
        except StopIteration:
            self.episodic_memory_iter = iter(self.episodic_memory_loader)
            batch = next(self.episodic_memory_iter)
        
        device = self.params['device']
        inp, targ, task_id, *_ = batch
        return inp.to(device), targ.to(device), task_id.to(device)

    def prepare_train_loader(self, task_id):
        num_workers = self.params.get('num_dataloader_workers', 0)
        return self.benchmark.load(task_id, self.params['batch_size_train'],
                                   num_workers=num_workers, pin_memory=True)[0]

    def training_step(self, task_ids, inp, targ, optimizer, criterion):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_ids)
        loss = criterion(pred, targ)
        loss.backward()
        optimizer.step()

    def training_step(self, task_ids, inp, targ, optimizer, criterion, sample_weight=None, sensitive=None):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_ids)
        criterion.reduction = "none"
        loss = criterion(pred, targ)
        criterion.reduction = "mean"
        if sample_weight is not None:
            loss = loss*sample_weight
            # print(f"{loss=}")
            # print(f"{loss.shape=}")
            # print(f"{sample_weight.shape=}")
        loss = loss.mean()
        loss.backward()
        if (task_ids[0] > 1) and self.params['tau']:
            grad_batch = flatten_grads(self.backbone).detach().clone()
            optimizer.zero_grad()

            # get grad_ref
            inp_ref, targ_ref, task_ids_ref = self.sample_batch_from_memory()
            pred_ref = self.backbone(inp_ref, task_ids_ref)
            loss = criterion(pred_ref, targ_ref.reshape(len(targ_ref)))
            loss.backward()
            grad_ref = flatten_grads(self.backbone).detach().clone()
            grad_batch += self.params['tau']*grad_ref

            optimizer.zero_grad()
            self.backbone = assign_grads(self.backbone, grad_batch)
        optimizer.step()

class BaseMemoryContinualAlgoritm(BaseContinualAlgoritm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_batch_from_memory(self):
        try:
            batch = next(self.episodic_memory_iter)
        except StopIteration:
            self.episodic_memory_iter = iter(self.episodic_memory_loader)
            batch = next(self.episodic_memory_iter)
        
        device = self.params['device']
        inp, targ, task_id, *_ = batch
        return inp.to(device), targ.to(device), task_id.to(device)

    def training_task_end(self):
        if self.requires_memory:
            self.update_episodic_memory()
        self.current_task += 1

    def update_episodic_memory(self):
        # called when training_task_end
        self.update_memory_after_train() # Do something after train
        self.episodic_memory_loader, _ = self.benchmark.load_memory_joint(self.current_task,
                                                                          batch_size=self.params['batch_size_memory'],
                                                                          shuffle=True,
                                                                          pin_memory=True)
        self.episodic_memory_iter = iter(self.episodic_memory_loader)

    def update_memory_after_train(self):
        pass

    # def load_memory_joint(self,
    #                       task: int,
    #                       batch_size: int,
    #                       shuffle: Optional[bool] = True,
    #                       num_workers: Optional[int] = 0,
    #                       pin_memory: Optional[bool] = True) -> Tuple[DataLoader, DataLoader]:
    #     if task > self.num_tasks:
    #         raise ValueError(f"Asked to load memory of task={task} but the benchmark has {self.num_tasks} tasks")
    #     trains, tests = [], []
    #     for t in range(1, task + 1):
    #         train_indices = self.memory_indices_train[t]
    #         test_indices = self.memory_indices_test[t]
    #         train_dataset = Subset(self.trains[t], train_indices)
    #         test_dataset = Subset(self.tests[t], test_indices)
    #         trains.append(train_dataset)
    #         tests.append(test_dataset)
    
    #     trains, tests = ConcatDataset(trains), ConcatDataset(tests)
    #     train_loader = DataLoader(trains, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)
    #     test_loader = DataLoader(tests, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)
    #     return train_loader, test_loader
