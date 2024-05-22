import torch
import numpy as np

from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.utils import flatten_grads, assign_grads

class BaselineContinualAlgoritm(ContinualAlgorithm):
    def sample_batch_from_memory(self):
        try:
            batch = next(self.episodic_memory_iter)
        except StopIteration:
            self.episodic_memory_iter = iter(self.episodic_memory_loader)
            batch = next(self.episodic_memory_iter)
        
        device = self.params['device']
        inp, targ, task_id, *_ = batch
        return inp.to(device), targ.to(device), task_id.to(device)

    def before_training_task(self):
        # called before loader, optimizer, criterion initialized
        pass

    def before_training_epoch(self):
        if hasattr(super(), "before_training_task"):
            super().before_training_task()
        self.weight_for_task = list()
        self.classwise_mean_grad = list()


    def prepare_train_loader(self, task_id, epoch=None):
        num_workers = self.params.get('num_dataloader_workers', torch.get_num_threads())
        return self.benchmark.load(task_id, self.params['batch_size_train'],
                                   num_workers=num_workers, pin_memory=True)[0]

    def training_step(self, task_ids, inp, targ, optimizer, criterion):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_ids)
        loss = criterion(pred, targ)
        loss.backward()
        optimizer.step()

    def training_step(self, task_ids, inp, targ, optimizer, criterion, sample_weight=None, sensitive_label=None):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_ids)
        criterion.reduction = "none"
        loss = criterion(pred, targ)
        criterion.reduction = "mean"
        if sample_weight is not None:
            loss = loss*sample_weight
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

class BaseMemoryContinualAlgoritm(BaselineContinualAlgoritm):
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
        print("training_task_end")
        super().training_task_end()
        # if self.requires_memory:
        #     self.update_episodic_memory()
        # self.current_task += 1

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
