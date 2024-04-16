import torch
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.utils import flatten_grads, assign_grads
from torch.nn.functional import relu, avg_pool2d
import matplotlib.pyplot as plt
import time

from algorithms.base import Heuristic
from algorithms.optimization import LS_solver, absolute_minsum_LP_solver, absolute_and_nonabsolute_minsum_LP_solver

import copy
import os

class Heuristic2(Heuristic):
    def get_loss_grad(self, task_id, loader, current_set = False):
        criterion = self.prepare_criterion(task_id)
        device = self.params['device']
        inc_num = self.benchmark.num_classes_per_split # MNIST
        if current_set:
            classwise_loss = {x:list() for x in self.benchmark.class_idx[(task_id-1)*inc_num:task_id*inc_num]}
            classwise_grad = {x:list() for x in self.benchmark.class_idx[(task_id-1)*inc_num:task_id*inc_num]}
        else:
            classwise_loss = {x:list() for x in self.benchmark.class_idx[:(task_id-1)*inc_num]}
            classwise_grad = {x:list() for x in self.benchmark.class_idx[:(task_id-1)*inc_num]}
        new_grads, grads = None, None
        
        for batch_idx, (inp, targ, t_id, *_) in enumerate(loader):
            # self.backbone.forward
            inp, targ, t_id  = inp.to(device), targ.to(device), t_id.to(device)
            pred, embeds = self.backbone.forward_embeds(inp)
            self.pred_shape = pred.shape[1]
            self.embeds_shape = embeds.shape[1]
            criterion.reduction = "none"
            loss = criterion(pred, targ.reshape(-1))
            criterion.reduction = "mean"
            
            bias_grads = torch.autograd.grad(loss.mean(), pred)[0]
            bias_expand = torch.repeat_interleave(bias_grads, embeds.shape[1], dim=1)
            weight_grads = bias_expand * embeds.repeat(1, pred.shape[1])
            grads = torch.cat([bias_grads, weight_grads], dim=1)
            grads = grads.cpu()

            for i, e in enumerate(targ):
                classwise_loss[e.cpu().item()].append(loss[i].detach().cpu()) # to prevent memory overflow
                classwise_grad[e.cpu().item()].append(grads[i].detach().cpu())

            if current_set:
                new_grads = grads if new_grads is None else torch.cat([new_grads, grads])

            self.backbone.zero_grad()
            
        return classwise_loss, classwise_grad, new_grads
    
    def get_loss_grad_all(self, task_id):
        num_workers = self.params.get('num_dataloader_workers', 0)

        classwise_loss, classwise_grad, _ = self.get_loss_grad(task_id, self.episodic_memory_loader, current_set = False)
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=False,
                                   num_workers=num_workers, pin_memory=True)[0]
        current_loss, current_grad, new_grads = self.get_loss_grad(task_id, train_loader, current_set = True)
        r_new_grads = new_grads[self.non_select_indexes]
        # r_new_grads = new_grads
        classwise_loss.update(current_loss)
        classwise_grad.update(current_grad)

        losses = []
        grads = []
        for k, v in classwise_loss.items():
            v3 = classwise_grad[k]
            # loss_ = torch.stack(v).mean(dim=0).view(1, -1).detach().clone()
            # grads_ = torch.stack(v3).mean(dim=0).view(1, -1).detach().clone()
            loss_ = torch.stack(v).mean(dim=0).view(1, -1)
            grads_ = torch.stack(v3).mean(dim=0).view(1, -1)
            losses.append(loss_)
            grads.append(grads_)

        with torch.no_grad():
            losses = torch.cat(losses, dim=0).view(1,-1)
            grads_all = torch.cat(grads, dim=0)
            
            # class별로 변화량이 비슷하도록 normalize
            n_grads_all = F.normalize(grads_all, p=2, dim=1) # (num_class) * (weight&bias 차원수)
            n_r_new_grads = F.normalize(r_new_grads, p=2, dim=1) # (후보수) * (weight&bias 차원수)

        return losses, n_grads_all, n_r_new_grads

    def converter_LS(self, losses, alpha, grads_all, new_grads, task=None):
        losses = torch.transpose(losses, 0, 1)
        grads_all = torch.transpose(grads_all, 0, 1) # (weight&bias 차원수) * (num_class)
        
        n = len(losses)
        m, dim = new_grads.shape

        c = torch.zeros_like(losses)
        d = torch.zeros_like(grads_all)

        for j in range(n):
            c[j] = n*losses[j] - losses.sum()
            d[:,j] = n*grads_all[:,j] - grads_all.sum(axis=1)

        d = alpha*d
        dg = torch.matmul(d.T, new_grads.T)
        dg, c = dg/(n**2), c/(n**2)
        return dg, c

    def converter_LP_absolute_additional(self, losses, alpha, grads_all, new_grads, task=None):
        """
        for LP with linear additional term
        return A, b, C, d
        where A, b are linear coefficient for absolute term, C, d are non-absolute term
        """
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        losses = torch.transpose(losses, 0, 1)
        grads_all = torch.transpose(grads_all, 0, 1) # (weight&bias 차원수) * (num_class)
        
        n = len(losses)
        m, dim = new_grads.shape

        c = torch.zeros_like(losses, device=device)
        d = torch.zeros_like(grads_all, device=device)

        lc = torch.zeros([n, 1], device=device)
        ld = torch.zeros([grads_all.shape[0], n], device=device)

        for j in range(n):
            c[j] = n*losses[j] - losses.sum()
            d[:,j] = n*grads_all[:,j] - grads_all.sum(axis=1)

            lc[j] = losses[j]
            ld[:,j] = grads_all[:,j]

        d = alpha*d
        dg = torch.matmul(d.T, new_grads.T)

        dg, c = dg/(n**2), c/(n**2)

        lmbd = self.params.get('lambda', 0.0)
        lmbd_old = self.params.get('lambda_old', 0.0)

        ld = alpha*ld
        ldg = torch.matmul(ld.T, new_grads.T)

        ldg, lc = lmbd*ldg, lmbd*lc

        ldg[:n-num_current_classes] = lmbd_old*ldg[:n-num_current_classes]/(n-num_current_classes)
        ldg[n-num_current_classes:] = ldg[n-num_current_classes:]/num_current_classes
        lc[:n-num_current_classes] = lmbd_old*lc[:n-num_current_classes]/(n-num_current_classes)
        lc[n-num_current_classes:] = lc[n-num_current_classes:]/num_current_classes
        return dg, c, ldg, lc

    def converter_LP_absolute_only(self, losses, alpha, grads_all, new_grads, task=None):
        A, b, C, d = self.converter_LP_absolute_additional(losses, alpha, grads_all, new_grads, task=task)
        return torch.concatenate([A, C], axis=0), torch.concatenate([b, d], axis=0)

    def prepare_train_loader(self, task_id, epoch=0):
        solver = self.params.get('solver')
        if (solver is None) or ("absolute_and_nonabsolute" in solver and "LP" in solver):
            solver = absolute_and_nonabsolute_minsum_LP_solver
            self.converter = self.converter_LP_absolute_additional
        elif "absolute" in solver and "LP" in solver:
            solver = absolute_minsum_LP_solver
            self.converter = self.converter_LP_absolute_only
        elif "LS" in solver:
            solver = LS_solver
            self.converter = self.converter_LS
        else:
            raise NotImplementedError

        # print(f"{solver=}")
        # print(f"{self.converter=}")
        return super().prepare_train_loader(task_id, solver=solver, epoch=epoch)


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
            inp_ref, targ_ref, task_ids_ref, sample_weight_ref = self.sample_batch_from_memory()
            pred_ref = self.backbone(inp_ref, task_ids_ref)
            loss = criterion(pred_ref, targ_ref.reshape(len(targ_ref)))
            loss.backward()
            grad_ref = flatten_grads(self.backbone).detach().clone()
            grad_batch += self.params['tau']*grad_ref

            optimizer.zero_grad()
            self.backbone = assign_grads(self.backbone, grad_batch)
        optimizer.step()
