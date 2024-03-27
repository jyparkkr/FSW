import torch
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.agem import AGEM
from cl_gym.algorithms.utils import flatten_grads, assign_grads
from torch.nn.functional import relu, avg_pool2d
from algorithms.base import Heuristic
from algorithms.optimization import minimax_LP_solver, minsum_LP_solver

import matplotlib.pyplot as plt

import time
import copy
import os


class Heuristic3(Heuristic):
    def sample_batch_from_memory(self):
        try:
            batch = next(self.episodic_memory_iter)
        except StopIteration:
            self.episodic_memory_iter = iter(self.episodic_memory_loader)
            batch = next(self.episodic_memory_iter)
        
        device = self.params['device']
        inp, targ, task_id, indices, sample_weight, sensitive_label, *_ = batch
        return inp.to(device), targ.to(device), task_id.to(device), indices, sample_weight.to(device), sensitive_label.to(device)

    def get_loss_grad(self, task_id, loader, current_set = False):
        criterion = self.prepare_criterion(task_id)
        device = self.params['device']
        inc_num = self.benchmark.num_classes_per_split # MNIST
        if current_set:
            classwise_loss_s0 = {x:list() for x in self.benchmark.class_idx[(task_id-1)*inc_num:task_id*inc_num]}
            classwise_grad_s0 = {x:list() for x in self.benchmark.class_idx[(task_id-1)*inc_num:task_id*inc_num]}
            classwise_loss_s1 = {x:list() for x in self.benchmark.class_idx[(task_id-1)*inc_num:task_id*inc_num]}
            classwise_grad_s1 = {x:list() for x in self.benchmark.class_idx[(task_id-1)*inc_num:task_id*inc_num]}
        else:
            classwise_loss_s0 = {x:list() for x in self.benchmark.class_idx[:(task_id-1)*inc_num]}
            classwise_grad_s0 = {x:list() for x in self.benchmark.class_idx[:(task_id-1)*inc_num]}
            classwise_loss_s1 = {x:list() for x in self.benchmark.class_idx[:(task_id-1)*inc_num]}
            classwise_grad_s1 = {x:list() for x in self.benchmark.class_idx[:(task_id-1)*inc_num]}
        # sensitive_loss = {x:list() for x in range(2)}
        # sensitive_grad_dict = {x:list() for x in range(2)}
        new_grads, grads = None, None
        
        for batch_idx, items in enumerate(loader):
            inp, targ, t_id, indices, sample_weight, sensitive_label, *_ = items
            # self.backbone.forward
            inp, targ, t_id, sensitive_label  = inp.to(device), targ.to(device), t_id.to(device), sensitive_label.to(device)
            pred, embeds = self.forward_embeds(inp)
            self.pred_shape = pred.shape[1]
            self.embeds_shape = embeds.shape[1]
            criterion.reduction = "none"
            loss = criterion(pred, targ.reshape(-1))
            criterion.reduction = "mean"
            
            bias_grads = torch.autograd.grad(loss.mean(), pred)[0]
            bias_expand = torch.repeat_interleave(bias_grads, embeds.shape[1], dim=1)
            weight_grads = bias_expand * embeds.repeat(1, pred.shape[1])
            grads = torch.cat([bias_grads, weight_grads], dim=1)
            sensitive = torch.ne(targ, sensitive_label)
            sensitive = sensitive.long()

            grads = grads.cpu()

            for i, e in enumerate(targ):
                if sensitive[i].cpu().item() == 0:
                    classwise_loss_s0[e.cpu().item()].append(loss[i].detach().cpu())
                    classwise_grad_s0[e.cpu().item()].append(grads[i].detach().cpu())
                elif sensitive[i].cpu().item() == 1:
                    classwise_loss_s1[e.cpu().item()].append(loss[i].detach().cpu())
                    classwise_grad_s1[e.cpu().item()].append(grads[i].detach().cpu())
                else:
                    raise NotImplementedError

            if current_set:
                new_grads = grads if new_grads is None else torch.cat([new_grads, grads])

            self.backbone.zero_grad()

        classwise_loss_all = [classwise_loss_s0, classwise_loss_s1]
        classwise_grad_all = [classwise_grad_s0, classwise_grad_s1]
        return classwise_loss_all, classwise_grad_all, new_grads
    
    def get_loss_grad_all(self, task_id):
        num_workers = self.params.get('num_dataloader_workers', 0)

        classwise_loss_all, classwise_grad_all, _ = self.get_loss_grad(task_id, self.episodic_memory_loader, current_set = False)
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=False,
                                   num_workers=num_workers, pin_memory=True)[0]
        current_loss_all, current_grad_all, new_grads = self.get_loss_grad(task_id, train_loader, current_set = True)
        r_new_grads = new_grads[self.non_select_indexes]

        classwise_loss_all[0].update(current_loss_all[0])
        classwise_loss_all[1].update(current_loss_all[1])
        classwise_grad_all[0].update(current_grad_all[0])
        classwise_grad_all[1].update(current_grad_all[1])

        losses = []
        grads = []
        for i, classwise_loss in enumerate(classwise_loss_all):
            for k, v in classwise_loss.items():
                v3 = classwise_grad_all[i][k]
                if len(v) == 0 :
                    print(f"###classwise_loss of {k}, s={i} is missing###")
                    raise NotImplementedError

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

    def converter(self, losses, alpha, grads_all, new_grads):
        """
        losses, grads_all의 위의 절반은 s=0, 아래 절반은 s=1임. transpose 후 각각의 difference를 return해야 함.
        """
        device = 'cpu'
        losses = torch.transpose(losses, 0, 1)
        grads_all = torch.transpose(grads_all, 0, 1) # (weight&bias 차원수) * (num_class)
        
        n = len(losses)//2
        # m, dim = new_grads.shape

        c = torch.zeros([n, 1], device=device)
        d = torch.zeros([grads_all.shape[0], n], device=device)

        lc = torch.zeros([n, 1], device=device)
        ld = torch.zeros([grads_all.shape[0], n], device=device)

        for j in range(n):
            c[j] = losses[j] - losses[n+j]
            d[:,j] = grads_all[:,j] - grads_all[:,n+j]

            lc[j] = (losses[j] + losses[n+j])/2
            ld[:,j] = (grads_all[:,j] + grads_all[:,n+j])/2

        d = alpha*d
        dg = torch.matmul(d.T, new_grads.T)

        dg, c = dg/n, c/n

        lmbd = self.params.get('lambda', 0.0)
        lmbd_old = self.params.get('lambda_old', 0.0)

        ld = alpha*ld
        ldg = torch.matmul(ld.T, new_grads.T)

        ldg, lc = lmbd*ldg, lmbd*lc

        ldg[:n-2], ldg[n-2:] = lmbd_old*ldg[:n-2]/(n-2), ldg[n-2:]/2
        lc[:n-2], lc[n-2:] = lmbd_old*lc[:n-2]/(n-2), lc[n-2:]/2

        return torch.concatenate([dg, ldg], axis=0), torch.concatenate([c, lc], axis=0)
        
    def converter_old(self, losses, alpha, grads_all, new_grads):
        """
        losses, grads_all의 위의 절반은 s=0, 아래 절반은 s=1임. transpose 후 각각의 difference를 return해야 함.
        """
        device = 'cpu'
        losses = torch.transpose(losses, 0, 1)
        grads_all = torch.transpose(grads_all, 0, 1) # (weight&bias 차원수) * (num_class)
        
        n = len(losses)//2
        # m, dim = new_grads.shape

        c = torch.zeros([n, 1], device=device)
        d = torch.zeros([grads_all.shape[0], n], device=device)

        lc = torch.zeros([n, 1], device=device)
        ld = torch.zeros([grads_all.shape[0], n], device=device)

        for j in range(n):
            c[j] = losses[j] - losses[n+j]
            d[:,j] = grads_all[:,j] - grads_all[:,n+j]

            lc[j] = (losses[j] + losses[n+j])/2
            ld[:,j] = (grads_all[:,j] + grads_all[:,n+j])/2

        d = alpha*d
        dg = torch.matmul(d.T, new_grads.T)

        dg, c = dg/n, c/n
        return dg, c

    def prepare_train_loader(self, task_id, epoch=0):
        solver = self.params.get('solver')
        if solver is None:
            agg = self.params['fairness_agg']
            if agg == "mean":
                solver = minsum_LP_solver
            elif agg == "max":
                solver = minimax_LP_solver
            else:
                raise NotImplementedError
            
        return super().prepare_train_loader(task_id, solver=solver, epoch=epoch)

    def training_step(self, task_ids, inp, targ, optimizer, criterion, sample_weight=None, sensitive_label=None):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_ids)
        criterion.reduction = "none"
        loss = criterion(pred, targ)
        criterion.reduction = "mean"
        if sample_weight is not None:
            loss = loss*sample_weight
            # print(f"{loss.shape=}")
            # print(f"{sample_weight.shape=}")
        loss = loss.mean()
        loss.backward()
        if (task_ids[0] > 1) and self.params['tau']:
            grad_batch = flatten_grads(self.backbone).detach().clone()
            optimizer.zero_grad()

            # get grad_ref
            inp_ref, targ_ref, task_ids_ref, *_ = self.sample_batch_from_memory()
            pred_ref = self.backbone(inp_ref, task_ids_ref)
            loss = criterion(pred_ref, targ_ref.reshape(len(targ_ref)))
            loss.backward()
            grad_ref = flatten_grads(self.backbone).detach().clone()
            grad_batch += self.params['tau']*grad_ref

            optimizer.zero_grad()
            self.backbone = assign_grads(self.backbone, grad_batch)
        optimizer.step()
