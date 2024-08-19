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
from algorithms.optimization import absolute_minimax_LP_solver, absolute_minsum_LP_solver, absolute_and_nonabsolute_minsum_LP_solver
from algorithms.optimization.scipy_solver import absolute_minsum_LP_solver_v3

import matplotlib.pyplot as plt

import time
import copy
import os

class Heuristic3(Heuristic):
    def __init__(self, backbone, benchmark, params, **kwargs):
        super().__init__(backbone, benchmark, params, **kwargs)
        self.absolute_minimax_LP_solver = absolute_minimax_LP_solver
        self.absolute_minsum_LP_solver = absolute_minsum_LP_solver
        self.absolute_minsum_LP_solver_v3 = absolute_minsum_LP_solver_v3
        self.absolute_and_nonabsolute_minsum_LP_solver = absolute_and_nonabsolute_minsum_LP_solver

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

        loaded_batch = list()
        for batch_idx, items in enumerate(loader):
            inp, targ, t_id, indices, sample_weight, sensitive_label, *_ = items
            if current_set:
                loaded_batch.append(items)
            # self.backbone.forward
            inp, targ, t_id, sensitive_label  = inp.to(device), targ.to(device), t_id.to(device), sensitive_label.to(device)
            pred, embeds = self.backbone.forward_embeds(inp, t_id)
            self.pred_shape = pred.shape[1]
            self.embeds_shape = embeds.shape[1]
            criterion.reduction = "none"
            loss = criterion(pred, targ.reshape(-1))
            criterion.reduction = "mean"
            
            bias_grads = torch.autograd.grad(loss.mean(), pred)[0]
            bias_expand = torch.repeat_interleave(bias_grads, embeds.shape[1], dim=1)
            weight_grads = bias_expand * embeds.repeat(1, pred.shape[1])
            grads = torch.cat([bias_grads, weight_grads], dim=1)
            if self.params['dataset'] == "BiasedMNIST":
                sensitive = torch.ne(targ, sensitive_label)
            else:
                sensitive = sensitive_label
            sensitive = sensitive.long()
            grads = grads.cpu()

            for i, e in enumerate(targ):
                if not current_set and e >= (task_id-1)*inc_num:
                    continue
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
        return classwise_loss_all, classwise_grad_all, new_grads, loaded_batch

    def get_loss_grad_all(self, task_id):
        num_workers = self.params.get('num_dataloader_workers', torch.get_num_threads())

        classwise_loss_all, classwise_grad_all, *_ = self.get_loss_grad(task_id, self.episodic_memory_loader, current_set = False)
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=False,
                                   num_workers=num_workers, pin_memory=True)[0]
        current_loss_all, current_grad_all, new_grads, new_batch = self.get_loss_grad(task_id, train_loader, current_set = True)
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
            self.classwise_mean_grad.append(torch.norm(grads_all, dim=1))

            # normalize to make class/group difference be similar
            if self.params.get('no_class_grad_norm', False):
                n_grads_all = grads_all
            else:
                n_grads_all = F.normalize(grads_all, p=2, dim=1) # (num_class) * (weight&bias 차원수)
            if self.params.get('no_datapoints_grad_norm', False):
                n_r_new_grads = r_new_grads
            else:
                n_r_new_grads = F.normalize(r_new_grads, p=2, dim=1) # (후보수) * (weight&bias 차원수)

        return losses, n_grads_all, n_r_new_grads, new_batch
    

    def get_loss_grad_model(self, task_id, loader, current_set = False, model=None):
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

        loaded_batch = list()
        for batch_idx, items in enumerate(loader):
            inp, targ, t_id, indices, sample_weight, sensitive_label, *_ = items
            if current_set:
                loaded_batch.append(items)
            # self.backbone.forward
            inp, targ, t_id, sensitive_label  = inp.to(device), targ.to(device), t_id.to(device), sensitive_label.to(device)
            pred, embeds = model.forward_embeds(inp, t_id)
            self.pred_shape = pred.shape[1]
            self.embeds_shape = embeds.shape[1]
            criterion.reduction = "none"
            loss = criterion(pred, targ.reshape(-1))
            criterion.reduction = "mean"
            
            bias_grads = torch.autograd.grad(loss.mean(), pred)[0]
            bias_expand = torch.repeat_interleave(bias_grads, embeds.shape[1], dim=1)
            weight_grads = bias_expand * embeds.repeat(1, pred.shape[1])
            grads = torch.cat([bias_grads, weight_grads], dim=1)
            if self.params['dataset'] == "BiasedMNIST":
                sensitive = torch.ne(targ, sensitive_label)
            else:
                sensitive = sensitive_label
            sensitive = sensitive.long()
            grads = grads.cpu()

            for i, e in enumerate(targ):
                if not current_set and e >= (task_id-1)*inc_num:
                    continue
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

            model.zero_grad()

        classwise_loss_all = [classwise_loss_s0, classwise_loss_s1]
        classwise_grad_all = [classwise_grad_s0, classwise_grad_s1]
        return classwise_loss_all, classwise_grad_all, new_grads, loaded_batch


    def measure_loss(self, task_id, model):
        num_workers = self.params.get('num_dataloader_workers', torch.get_num_threads())

        classwise_loss_all, classwise_grad_all, *_ = self.get_loss_grad_model(task_id, self.episodic_memory_loader, model=model, current_set = False)
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=False,
                                   num_workers=num_workers, pin_memory=True)[0]
        current_loss_all, current_grad_all, new_grads, new_batch = self.get_loss_grad_model(task_id, train_loader, model=model, current_set = True)
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
        return losses


    def converter_LP_absolute_additional_EO(self, losses, alpha, grads_all, new_grads, task=None):
        """
        LP with linear additional term
        input: 
            losses: torch.cat([prev_classwise_loss_s0, prev_classwise_loss_s1])
            alpha: coefficient for gradients
            grads_all: torch.cat([prev_classwise_gradient_s0, prev_classwise_gradient_s1])
            new_grads: current data pointwise gradient
        output: 
            A: averaged alpha*(prev_classwise_gradient_s0 - prev_classwise_gradient_s1)·pointwise_gradient
            b: averaged prev_classwise_loss_s0 - prev_classwise_loss_s1
            C: averaged alpha*classwise_gradient·pointwise_gradient
            d: averaged prev_classwise_loss
        where A, b are linear coefficient for absolute term, C, d are non-absolute term
        min_x |b - Ax| + (d - Cx)
        """
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        losses = torch.transpose(losses, 0, 1) # (num_class) * (1)
        grads_all = torch.transpose(grads_all, 0, 1) # (num_class) * (weight&bias 차원수)
        
        n = len(losses)//2
        # m, dim = new_grads.shape

        c = torch.zeros([n, 1], device=device)
        d = torch.zeros([grads_all.shape[0], n], device=device)

        lc = torch.zeros([n, 1], device=device)
        ld = torch.zeros([grads_all.shape[0], n], device=device)

        for j in range(n):
            c[j] = (losses[j] - losses[n+j])/2  # |Z|=2
            d[:,j] = (grads_all[:,j] - grads_all[:,n+j])/2  # |Z|=2

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

        ldg[:n-num_current_classes] = lmbd_old*ldg[:n-num_current_classes]/(n-num_current_classes)
        ldg[n-num_current_classes:] = ldg[n-num_current_classes:]/num_current_classes
        lc[:n-num_current_classes] = lmbd_old*lc[:n-num_current_classes]/(n-num_current_classes)
        lc[n-num_current_classes:] = lc[n-num_current_classes:]/num_current_classes
        return dg, c, ldg, lc

    def converter_LP_absolute_additional_DP(self, losses, alpha, grads_all, new_grads, task=None):
        """
        LP with linear additional term
        input: 
            losses: torch.cat([prev_classwise_loss_s0, prev_classwise_loss_s1])
            alpha: coefficient for gradients
            grads_all: torch.cat([prev_classwise_gradient_s0, prev_classwise_gradient_s1])
            new_grads: current data pointwise gradient
        output: 
            A: averaged alpha*(prev_classwise_gradient_s0 - prev_classwise_gradient_s1)·pointwise_gradient
            b: averaged prev_classwise_loss_s0 - prev_classwise_loss_s1
            C: averaged alpha*classwise_gradient·pointwise_gradient
            d: averaged prev_classwise_loss
        where A, b are linear coefficient for absolute term, C, d are non-absolute term
        min_x |b - Ax| + (d - Cx)
        """
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        losses = torch.transpose(losses, 0, 1) # (num_class) * (1)
        grads_all = torch.transpose(grads_all, 0, 1) # (num_class) * (weight&bias 차원수)
        
        n = len(losses)//2
        def m_y_z(y, z):
            return self.benchmark.m_dict[z][y]
        def m_z(z):
            return np.sum([m_y_z(y, z) for y in self.benchmark.class_idx])

        # m, dim = new_grads.shape

        c = torch.zeros([n, 1], device=device)
        d = torch.zeros([grads_all.shape[0], n], device=device)

        lc = torch.zeros([n, 1], device=device)
        ld = torch.zeros([grads_all.shape[0], n], device=device)

        for j in range(n):
            c[j] = (m_y_z(j, 0) / m_z(0) * losses[j] - m_y_z(j, 1) / m_z(1) * losses[n+j])/2 # |Z|=2
            d[:,j] = (m_y_z(j, 0) / m_z(0) * grads_all[:,j] - m_y_z(j, 1) / m_z(1) * grads_all[:,n+j])/2

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

        ldg[:n-num_current_classes] = lmbd_old*ldg[:n-num_current_classes]/(n-num_current_classes)
        ldg[n-num_current_classes:] = ldg[n-num_current_classes:]/num_current_classes
        lc[:n-num_current_classes] = lmbd_old*lc[:n-num_current_classes]/(n-num_current_classes)
        lc[n-num_current_classes:] = lc[n-num_current_classes:]/num_current_classes
        return dg, c, ldg, lc

    def converter_LP_absolute_only_EO(self, losses, alpha, grads_all, new_grads, task=None):
        """
        losses, grads_all의 위의 절반은 s=0, 아래 절반은 s=1임. transpose 후 각각의 difference를 return해야 함.
        only avilable when number of sensitive class is 2
        """
        A, b, C, d = self.converter_LP_absolute_additional_EO(losses, alpha, grads_all, new_grads, task=task)
        return torch.concatenate([A, C], axis=0), torch.concatenate([b, d], axis=0)

    def converter_LP_no_losses(self, losses, alpha, grads_all, new_grads, task=None):
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
        metric = self.params.get('metric')
        agg = self.params.get('fairness_agg')
        if solver is None:
            if metric == "EO" or metric is None:
                if agg == "mean" or agg is None:
                    solver = absolute_and_nonabsolute_minsum_LP_solver
                    self.converter = self.converter_LP_absolute_additional_EO
                elif agg == "max":
                    raise NotImplementedError
                    solver = absolute_minimax_LP_solver
                    self.converter = self.converter_LP_absolute_only_EO
                else:
                    raise NotImplementedError
            elif metric == "DP":
                if agg == "mean" or agg is None:
                    solver = absolute_and_nonabsolute_minsum_LP_solver
                    self.converter = self.converter_LP_absolute_additional_DP
                else:
                    raise NotImplementedError
            elif metric == "no_metrics":
                pass
            else:
                raise NotImplementedError
        else:
            solver = getattr(self, solver)
            self.converter = getattr(self, self.params.get('converter'))    
        # print(f"{solver=}")
        # print(f"{self.converter=}")
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
