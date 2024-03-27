import torch
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import ContinualAlgorithm
import time

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
