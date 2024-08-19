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
    def get_loss_grad(self, task_id, loader, current_set = False, return_grad=False, configs=None):
        criterion = self.prepare_criterion(task_id)
        device = self.params['device']
        inc_num = self.benchmark.num_classes_per_split # MNIST
        params_dim = 0
        for name, params in self.backbone.named_parameters():
            if not 'bn' in name and not 'IC' in name:
                # grads_j.append(params.grad.clone().detach().cpu().view(-1))
                params_dim += len(params.view(-1))
        if current_set:
            classwise_loss = {x:0 for x in self.benchmark.class_idx[(task_id-1)*inc_num:task_id*inc_num]}
            classwise_grad = {x:torch.zeros(params_dim) for x in self.benchmark.class_idx[(task_id-1)*inc_num:task_id*inc_num]}
            classwise_count = {x:0 for x in self.benchmark.class_idx[(task_id-1)*inc_num:task_id*inc_num]}
        else:
            classwise_loss = {x:0 for x in self.benchmark.class_idx[:(task_id-1)*inc_num]}
            classwise_grad = {x:torch.zeros(params_dim) for x in self.benchmark.class_idx[:(task_id-1)*inc_num]}
            classwise_count = {x:0 for x in self.benchmark.class_idx[:(task_id-1)*inc_num]}

        new_grads, grads = None, None
        accmulated_dg, accmulated_ldg = None, None
        
        loaded_batch = list()
        self.backbone.eval()
        self.backbone.zero_grad()
        for batch_idx, items in enumerate(loader):
            # self.backbone.forward
            inp, targ, t_id, *_ = items
            if return_grad:
                loaded_batch.append(copy.deepcopy(items))
            inp, targ, t_id  = inp.to(device), targ.to(device), t_id.to(device)
            pred, embeds = self.backbone.forward_embeds(inp, t_id)
            self.pred_shape = pred.shape[1]
            self.embeds_shape = embeds.shape[1]
            criterion.reduction = "none"
            loss = criterion(pred, targ.reshape(-1))
            criterion.reduction = "mean"

            if self.params.get('all_layer_gradient', False):
                """
                This requires gradient to be matmul every batch.
                Otherwise needed memory would be larger
                """
                batch_size = inp.shape[0]
                grads = list()
                for j in range(batch_size):
                    loss[j].backward(retain_graph=True)
                    # parameters named_parameter로 loop 돌리기
                    grads_j = list()
                    for name, params in self.backbone.named_parameters():
                        if not 'bn' in name and not 'IC' in name:
                            # grads_j.append(params.grad.clone().detach().cpu().view(-1))
                            grads_j.append(params.grad.clone().detach().cpu().view(-1))
                    self.backbone.zero_grad()
                    grads_j = torch.cat(grads_j)
                    grads.append(grads_j)
                grads = torch.stack(grads, dim=0)
                # print(f"{batch_idx=}")
                if return_grad:
                    n_grads = F.normalize(grads, p=2, dim=1)
                    dg = torch.matmul(configs['d'].T, n_grads.T)
                    ldg = torch.matmul(configs['ld'].T, n_grads.T)

            else:
                bias_grads = torch.autograd.grad(loss.mean(), pred)[0]
                bias_expand = torch.repeat_interleave(bias_grads, embeds.shape[1], dim=1)
                weight_grads = bias_expand * embeds.repeat(1, pred.shape[1])
                grads = torch.cat([bias_grads, weight_grads], dim=1)
                grads = grads.clone().detach().cpu()

            for i, e in enumerate(targ):
                classwise_loss[e.cpu().item()] += loss[i].detach().cpu() # to prevent memory overflow
                # classwise_loss[e.cpu().item()].append(grads[i].detach().cpu()) # to prevent memory overflow
                classwise_grad[e.cpu().item()] += grads[i]
                classwise_count[e.cpu().item()] += 1

            if return_grad:
                if self.params.get('all_layer_gradient', False):
                    accmulated_dg = dg if accmulated_dg is None else torch.cat([accmulated_dg, dg], dim=1)
                    accmulated_ldg = ldg if accmulated_ldg is None else torch.cat([accmulated_ldg, ldg], dim=1)
                else:
                    new_grads = grads if new_grads is None else torch.cat([new_grads, grads])

            self.backbone.zero_grad()
        
        for k, v in classwise_count.items():
            classwise_loss[k] = (classwise_loss[k]/v).view(1, -1)
            classwise_grad[k] = (classwise_grad[k]/v).view(1, -1)
        
        if return_grad and self.params.get('all_layer_gradient', False):
            new_grads = [accmulated_dg, accmulated_ldg]

        return classwise_loss, classwise_grad, new_grads, loaded_batch
    
    def get_loss_grad_all(self, task_id):
        num_workers = self.params.get('num_dataloader_workers', torch.get_num_threads())
        classwise_loss, classwise_grad, *_ = self.get_loss_grad(task_id, self.episodic_memory_loader, current_set = False)
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=True,
                                   num_workers=num_workers, pin_memory=True)[0]
        current_loss, current_grad, *_ = self.get_loss_grad(task_id, train_loader, current_set = True)
        print(f"get_loss_grad done")
        classwise_loss.update(current_loss)
        classwise_grad.update(current_grad)

        losses = []
        grads = []
        for k, v in classwise_loss.items():
            v3 = classwise_grad[k]
            # loss_ = torch.stack(v).mean(dim=0).view(1, -1).detach().clone()
            # grads_ = torch.stack(v3).mean(dim=0).view(1, -1).detach().clone()
            losses.append(v)
            grads.append(v3)

        with torch.no_grad():
            losses = torch.cat(losses, dim=0).view(1,-1)
            grads_all = torch.cat(grads, dim=0)
            self.classwise_mean_grad.append(torch.norm(grads_all, dim=1)) # for plotting (debug)
            
            # class/group별로 변화량이 비슷하도록 normalize
            if self.params.get('no_class_grad_norm', False):
                n_grads_all = grads_all
            else:
                n_grads_all = F.normalize(grads_all, p=2, dim=1) # (num_class) * (weight&bias 차원수)
            # print(f"{n_grads_all.shape=}")

        configs = self.converter_LP_upper(losses, self.alpha, n_grads_all, task=task_id)
        *_, new_grads, new_batch = self.get_loss_grad(task_id, train_loader, current_set = True, return_grad= True, configs=configs)
        if self.params.get('all_layer_gradient', False):
            dg, ldg = new_grads
            configs['dg'] = dg
            configs['ldg'] = ldg
            n_new_grads = configs
        else:
            with torch.no_grad():
                if self.params.get('no_datapoints_grad_norm', False):
                    n_new_grads = new_grads
                else:
                    n_new_grads = F.normalize(new_grads, p=2, dim=1) # (후보수) * (weight&bias 차원수)
        return losses, n_grads_all, n_new_grads, new_batch

    def converter_LS(self, losses, alpha, grads_all, new_grads, task=None):
        """
        for LS
        return A, b
        where A, b are coefficient
        min_x (Ax - b)^2
        """

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

    def converter_LP_upper(self, losses, alpha, grads_all, task=None):
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        losses = torch.transpose(losses, 0, 1)
        grads_all = torch.transpose(grads_all, 0, 1) # (weight&bias 차원수) * (num_class)
        
        n = len(losses)

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
        ld = alpha*ld

        configs = dict()
        configs['c'], configs['d'], configs['lc'], configs['ld'] = c, d, lc, ld
        return configs

    def converter_LP_lower(self, configs, losses, alpha, task=None):
        losses = torch.transpose(losses, 0, 1)
        n = len(losses)
        num_current_classes = self.get_num_current_classes(task)

        dg = configs['dg']
        c = configs['c']

        dg, c = dg/(n**2), c/(n**2)

        lmbd = self.params.get('lambda', 0.0)
        lmbd_old = self.params.get('lambda_old', 0.0)

        ldg = configs['ldg']
        lc = configs['lc']

        ldg, lc = lmbd*ldg, lmbd*lc

        ldg[:n-num_current_classes] = lmbd_old*ldg[:n-num_current_classes]/(n-num_current_classes)
        ldg[n-num_current_classes:] = ldg[n-num_current_classes:]/num_current_classes
        lc[:n-num_current_classes] = lmbd_old*lc[:n-num_current_classes]/(n-num_current_classes)
        lc[n-num_current_classes:] = lc[n-num_current_classes:]/num_current_classes
        return dg, c, ldg, lc

    def converter_LP_absolute_additional(self, losses, alpha, grads_all, new_grads, task=None):
        """
        LP with linear additional term
        input: 
            losses: classwise loss
            alpha: coefficient for gradients
            grads_all: previous epoch classwise gradient
            new_grads: current data pointwise gradient
        output: 
            A: averaged alpha*(prev_classwise_gradient - avg_prev_classwise_gradient)·pointwise_gradient
            b: averaged prev_classwise_loss - avg_prev_classwise_loss
            C: averaged alpha*classwise_gradient·pointwise_gradient
            d: averaged prev_classwise_loss
        where A, b are linear coefficient for absolute term, C, d are non-absolute term
        min_x |b - Ax| + (d - Cx)
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

    # def prepare_train_loader(self, task_id, epoch=0):
    #     solver = self.params.get('solver')
    #     if (solver is None) or ("absolute_and_nonabsolute" in solver and "LP" in solver):
    #         solver = absolute_and_nonabsolute_minsum_LP_solver
    #         self.converter = self.converter_LP_absolute_additional
    #     elif "absolute" in solver and "LP" in solver:
    #         solver = absolute_minsum_LP_solver
    #         self.converter = self.converter_LP_absolute_only
    #     elif "LS" in solver:
    #         solver = LS_solver
    #         self.converter = self.converter_LS
    #     else:
    #         raise NotImplementedError

    #     # print(f"{solver=}")
    #     # print(f"{self.converter=}")
    #     return super().prepare_train_loader(task_id, solver=solver, epoch=epoch)

    def prepare_train_loader(self, task_id, epoch=0):
        solver = self.params.get('solver')
        metric = self.params.get('metric')
        agg = self.params.get('fairness_agg')
        if solver is None:
            if metric == "EER" or metric is None:
                if agg == "mean" or agg is None:
                    solver = absolute_and_nonabsolute_minsum_LP_solver
                    self.converter = self.converter_LP_absolute_additional
                else:
                    raise NotImplementedError
            elif metric == "no_metrics":
                pass
            else:
                raise NotImplementedError
        else:
            solver = getattr(self, solver)
            self.converter = getattr(self, self.params.get('converter'))    


    #     if (solver is None) or ("absolute_and_nonabsolute" in solver and "LP" in solver):
    #         solver = absolute_and_nonabsolute_minsum_LP_solver
    #         self.converter = self.converter_LP_absolute_additional
    #     elif "absolute" in solver and "LP" in solver:
    #         solver = absolute_minsum_LP_solver
    #         self.converter = self.converter_LP_absolute_only
    #     elif "LS" in solver:
    #         solver = LS_solver
    #         self.converter = self.converter_LS
    #     else:
    #         raise NotImplementedError

        # print(f"{solver=}")
        # print(f"{self.converter=}")
        return super().prepare_train_loader(task_id, solver=solver, epoch=epoch)

    def training_step(self, task_ids, inp, targ, optimizer, criterion, sample_weight=None, sensitive_label=None):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_ids)
        # pred = self.backbone(inp)
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