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
        grad_data, grads = None, None
        
        loaded_batch = list()
        self.backbone.eval()
        self.backbone.zero_grad()
        for batch_idx, items in enumerate(loader):
            # self.backbone.forward
            inp, targ, t_id, *_ = items
            if current_set:
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
                if current_set:
                    # print(f"{targ=}")
                    # print(f"{grads=}")
                    n_grads = F.normalize(grads, p=2, dim=1)
                    # print(f"{n_grads=}")

                # print(f"{batch_idx=}")
            else:
                bias_grads = torch.autograd.grad(loss.mean(), pred)[0]
                bias_expand = torch.repeat_interleave(bias_grads, embeds.shape[1], dim=1)
                weight_grads = bias_expand * embeds.repeat(1, pred.shape[1])
                grads = torch.cat([bias_grads, weight_grads], dim=1)
                grads = grads.clone().detach().cpu()

            for i, e in enumerate(targ):
                if not current_set and e >= (task_id-1)*inc_num:
                    continue
                classwise_loss[e.cpu().item()].append(loss[i].detach().cpu()) # to prevent memory overflow
                # classwise_loss[e.cpu().item()].append(grads[i].detach().cpu()) # to prevent memory overflow
                classwise_grad[e.cpu().item()].append(grads[i])

            grad_data = grads if grad_data is None else torch.cat([grad_data, grads])

            self.backbone.zero_grad()
            
        return classwise_loss, classwise_grad, grad_data, loaded_batch
    
    def get_loss_grad_all(self, task_id):
        num_workers = self.params.get('num_dataloader_workers', torch.get_num_threads())
        classwise_loss, classwise_grad, grad_data_prev, *_ = self.get_loss_grad(task_id, self.episodic_memory_loader, current_set = False)
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=True,
                                   num_workers=num_workers, pin_memory=True)[0]
        current_loss, current_grad, grad_data_current, new_batch = self.get_loss_grad(task_id, train_loader, current_set = True)
        classwise_loss.update(current_loss)
        classwise_grad.update(current_grad)

        loss_group = []
        grads = []
        for k, v in classwise_loss.items():
            v3 = classwise_grad[k]
            # loss_ = torch.stack(v).mean(dim=0).view(1, -1).detach().clone()
            # grads_ = torch.stack(v3).mean(dim=0).view(1, -1).detach().clone()
            loss_ = torch.stack(v).mean(dim=0).view(1, -1)
            grads_ = torch.stack(v3).mean(dim=0).view(1, -1)
            loss_group.append(loss_)
            grads.append(grads_)

        with torch.no_grad():
            loss_group = torch.cat(loss_group, dim=0).view(1,-1)
            grad_group = torch.cat(grads, dim=0)
            self.classwise_mean_grad.append(torch.norm(grad_group, dim=1))
            
            # class/group별로 변화량이 비슷하도록 normalize
            if self.params.get('no_class_grad_norm', False):
                grad_group = grad_group
            else:
                grad_group = F.normalize(grad_group, p=2, dim=1) # (num_class) * (weight&bias 차원수)
            if self.params.get('no_datapoints_grad_norm', False):
                grad_data_prev = grad_data_prev
                grad_data_current = grad_data_current
            else:
                grad_data_prev = F.normalize(grad_data_prev, p=2, dim=1) # (후보수) * (weight&bias 차원수)
                grad_data_current = F.normalize(grad_data_current, p=2, dim=1) # (후보수) * (weight&bias 차원수)

        return loss_group, grad_group, (grad_data_prev, grad_data_current), new_batch

    def get_loss_grad_model(self, task_id, loader, current_set = False, model=None):
        criterion = self.prepare_criterion(task_id)
        device = self.params['device']
        inc_num = self.benchmark.num_classes_per_split # MNIST
        if current_set:
            classwise_loss = {x:list() for x in self.benchmark.class_idx[(task_id-1)*inc_num:task_id*inc_num]}
            classwise_grad = {x:list() for x in self.benchmark.class_idx[(task_id-1)*inc_num:task_id*inc_num]}
        else:
            classwise_loss = {x:list() for x in self.benchmark.class_idx[:(task_id-1)*inc_num]}
            classwise_grad = {x:list() for x in self.benchmark.class_idx[:(task_id-1)*inc_num]}
        grad_data, grads = None, None
        
        model.eval()
        model.zero_grad()
        for batch_idx, items in enumerate(loader):
            inp, targ, t_id, *_ = items
            inp, targ, t_id  = inp.to(device), targ.to(device), t_id.to(device)
            pred, embeds = model.forward_embeds(inp, t_id)
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
                    for name, params in model.named_parameters():
                        if not 'bn' in name and not 'IC' in name:
                            # grads_j.append(params.grad.clone().detach().cpu().view(-1))
                            grads_j.append(params.grad.clone().detach().cpu().view(-1))
                    model.zero_grad()
                    grads_j = torch.cat(grads_j)
                    grads.append(grads_j)
                grads = torch.stack(grads, dim=0)
                if current_set:
                    print(f"{targ=}")
                    print(f"{grads=}")
                    n_grads = F.normalize(grads, p=2, dim=1)
                    print(f"{n_grads=}")

                # print(f"{batch_idx=}")
            else:
                bias_grads = torch.autograd.grad(loss.mean(), pred)[0]
                bias_expand = torch.repeat_interleave(bias_grads, embeds.shape[1], dim=1)
                weight_grads = bias_expand * embeds.repeat(1, pred.shape[1])
                grads = torch.cat([bias_grads, weight_grads], dim=1)
                grads = grads.clone().detach().cpu()

            for i, e in enumerate(targ):
                if not current_set and e >= (task_id-1)*inc_num:
                    continue
                classwise_loss[e.cpu().item()].append(loss[i].detach().cpu()) # to prevent memory overflow
                # classwise_loss[e.cpu().item()].append(grads[i].detach().cpu()) # to prevent memory overflow
                classwise_grad[e.cpu().item()].append(grads[i])

            model.zero_grad()
            
        return classwise_loss, classwise_grad


    def measure_loss(self, task_id, model):
        num_workers = self.params.get('num_dataloader_workers', torch.get_num_threads())
        classwise_loss, classwise_grad, *_ = self.get_loss_grad_model(task_id, self.episodic_memory_loader, model=model, current_set = False)
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=True,
                                   num_workers=num_workers, pin_memory=True)[0]
        current_loss, current_grad, *_ = self.get_loss_grad_model(task_id, train_loader, model=model, current_set = True)
        classwise_loss.update(current_loss)
        classwise_grad.update(current_grad)

        loss_group = []
        grads = []
        for k, v in classwise_loss.items():
            v3 = classwise_grad[k]
            loss_ = torch.stack(v).mean(dim=0).view(1, -1)
            grads_ = torch.stack(v3).mean(dim=0).view(1, -1)
            loss_group.append(loss_)
            grads.append(grads_)

        with torch.no_grad():
            loss_group = torch.cat(loss_group, dim=0).view(1,-1)

        return loss_group


    def converter_LS(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        """
        for LS
        return A, b
        where A, b are coefficient
        min_x (Ax - b)^2
        """

        loss_group = torch.transpose(loss_group, 0, 1)
        grad_group = torch.transpose(grad_group, 0, 1) # (weight&bias 차원수) * (num_class)
        
        n = len(loss_group)
        m, dim = grad_data.shape

        b = torch.zeros_like(loss_group)
        grad_diff = torch.zeros_like(grad_group)

        for j in range(n):
            b[j] = loss_group[j] - loss_group.mean()
            grad_diff[:,j] = grad_group[:,j] - grad_group.mean(axis=1)

        grad_diff = alpha*grad_diff
        A = torch.matmul(grad_diff.T, grad_data.T)
        A, b = A/n, b/n
        return A, b

    def converter_LP_absolute_additional(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        """
        LP with linear additional term
        input: 
            loss_group: classwise loss
            alpha: coefficient for gradients
            grad_group: previous epoch classwise gradient
            grad_data: current data pointwise gradient
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
        loss_group = torch.transpose(loss_group, 0, 1)
        grad_group = torch.transpose(grad_group, 0, 1) # (weight&bias 차원수) * (num_class)
        
        n = len(loss_group)
        m, dim = grad_data.shape

        b = torch.zeros_like(loss_group, device=device)
        grad_diff = torch.zeros_like(grad_group, device=device)

        d = torch.zeros([n, 1], device=device)
        classwise_grads = torch.zeros([grad_group.shape[0], n], device=device)

        for j in range(n):
            b[j] = loss_group[j] - loss_group.mean()
            grad_diff[:,j] = grad_group[:,j] - grad_group.mean(axis=1)

            d[j] = loss_group[j]
            classwise_grads[:,j] = grad_group[:,j]

        grad_diff = alpha*grad_diff
        A = torch.matmul(grad_diff.T, grad_data.T)

        A, b = A/n, b/n

        lmbd = self.params.get('lambda', 0.0)
        lmbd_old = self.params.get('lambda_old', 0.0)

        classwise_grads = alpha*classwise_grads
        C = torch.matmul(classwise_grads.T, grad_data.T)

        C, d = lmbd*C, lmbd*d

        C[:n-num_current_classes] = lmbd_old*C[:n-num_current_classes]/(n-num_current_classes)
        C[n-num_current_classes:] = C[n-num_current_classes:]/num_current_classes
        d[:n-num_current_classes] = lmbd_old*d[:n-num_current_classes]/(n-num_current_classes)
        d[n-num_current_classes:] = d[n-num_current_classes:]/num_current_classes
        return A, b, C, d

    def converter_LP_absolute_only(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        A, b, C, d = self.converter_LP_absolute_additional(loss_group, alpha, grad_group, grad_data, task=task)
        return torch.concatenate([A, C], axis=0), torch.concatenate([b, d], axis=0)

    def converter_LP_absolute_additional_v1(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        """
        LP with linear additional term
        input: 
            loss_group: classwise loss
            alpha: coefficient for gradients
            grad_group: previous epoch classwise gradient
            grad_data: current data pointwise gradient
            grad_data_prev: prev data pointwise gradient
        output: 
            A: averaged alpha*(prev_classwise_gradient - avg_prev_classwise_gradient)·pointwise_gradient
            b: (averaged prev_classwise_loss - avg_prev_classwise_loss) 
                - averaged tau*alpha*(prev_classwise_gradient - avg_prev_classwise_gradient)·buffer_pointwise_gradient
            C: averaged alpha*classwise_gradient·pointwise_gradient
            d: averaged prev_classwise_loss
                - averaged tau*alpha*classwise_gradient·buffer_pointwise_gradient
        where A, b are linear coefficient for absolute term, C, d are non-absolute term
        min_x |b - Ax| + (d - Cx)
        """
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        loss_group = torch.transpose(loss_group, 0, 1)
        grad_group = torch.transpose(grad_group, 0, 1) # (weight&bias 차원수) * (num_class)
        grad_data_prev = kwargs['grad_data_prev']
        tau = self.params['tau']
        
        n = len(loss_group)
        m, dim = grad_data.shape

        b = torch.zeros_like(loss_group, device=device)
        grad_diff = torch.zeros_like(grad_group, device=device)

        d = torch.zeros([n, 1], device=device)
        classwise_grads = torch.zeros([grad_group.shape[0], n], device=device)

        for j in range(n):
            b[j] = loss_group[j] - loss_group.mean()
            grad_diff[:,j] = grad_group[:,j] - grad_group.mean(axis=1)

            d[j] = loss_group[j]
            classwise_grads[:,j] = grad_group[:,j]

        grad_diff = alpha*grad_diff
        A = torch.matmul(grad_diff.T, grad_data.T)

        A, b = A/n, b/n
        b -= tau*torch.matmul(grad_diff.T, grad_data_prev.T.mean(dim=1, keepdim=True))

        lmbd = self.params.get('lambda', 0.0)
        lmbd_old = self.params.get('lambda_old', 0.0)

        classwise_grads = alpha*classwise_grads
        C = torch.matmul(classwise_grads.T, grad_data.T)

        C, d = lmbd*C, lmbd*d
        d -= tau*torch.matmul(classwise_grads.T, grad_data_prev.T.mean(dim=1, keepdim=True))

        C[:n-num_current_classes] = lmbd_old*C[:n-num_current_classes]/(n-num_current_classes)
        C[n-num_current_classes:] = C[n-num_current_classes:]/num_current_classes
        d[:n-num_current_classes] = lmbd_old*d[:n-num_current_classes]/(n-num_current_classes)
        d[n-num_current_classes:] = d[n-num_current_classes:]/num_current_classes

        return A, b, C, d


    def converter_LP_absolute_additional_v2(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        """
        For (absolute) loss difference calculation, only measure loss for past classes
        min_x |b - Ax| + (d - Cx)
        where A, b contain only pass classes
        """
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        loss_group = torch.transpose(loss_group, 0, 1)
        grad_group = torch.transpose(grad_group, 0, 1) # (weight&bias 차원수) * (num_class)
        
        n = len(loss_group)
        m, dim = grad_data.shape

        b = torch.zeros_like(loss_group, device=device)
        grad_diff = torch.zeros_like(grad_group, device=device)

        d = torch.zeros([n, 1], device=device)
        classwise_grads = torch.zeros([grad_group.shape[0], n], device=device)

        num_past_classes = n-num_current_classes
        for j in range(n):
            if not j >= num_past_classes:
                b[j] = loss_group[j] - loss_group[:num_past_classes].mean()
                grad_diff[:,j] = grad_group[:,j] - grad_group[:,:num_past_classes].mean(axis=1)

            d[j] = loss_group[j]
            classwise_grads[:,j] = grad_group[:,j]

        grad_diff = alpha*grad_diff
        A = torch.matmul(grad_diff.T, grad_data.T)

        A, b = A/num_past_classes, b/num_past_classes

        lmbd = self.params.get('lambda', 0.0)
        lmbd_old = self.params.get('lambda_old', 0.0)

        classwise_grads = alpha*classwise_grads
        C = torch.matmul(classwise_grads.T, grad_data.T)

        C, d = lmbd*C, lmbd*d

        C[:n-num_current_classes] = lmbd_old*C[:n-num_current_classes]/(n-num_current_classes)
        C[n-num_current_classes:] = C[n-num_current_classes:]/num_current_classes
        d[:n-num_current_classes] = lmbd_old*d[:n-num_current_classes]/(n-num_current_classes)
        d[n-num_current_classes:] = d[n-num_current_classes:]/num_current_classes
        return A, b, C, d



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
            self.converter = self.params.get('converter')


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