import torch
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.agem import AGEM
from cl_gym.algorithms.utils import flatten_grads, assign_grads
from torch.nn.functional import relu, avg_pool2d
from algorithms.base import Heuristic, hinge
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
        if isinstance(inp, list):
            inp = [x.to(device) for x in inp]
        else:
            inp = inp.to(device)
        return inp, targ.to(device), task_id.to(device), indices, sample_weight.to(device), sensitive_label.to(device)

    def before_training_epoch(self):
        ## debug - to compare btw 0-1 loss and CE loss
        self.class_acc = dict()
        self.class_acc_s0 = dict()
        self.class_acc_s1 = dict()

        self.class_loss = dict()
        self.class_loss_s0 = dict()
        self.class_loss_s1 = dict()

        self.class_pred_count = dict()
        self.class_pred_count_s0 = dict()
        self.class_pred_count_s1 = dict()
        self.count, self.count_s0, self.count_s1 = 0, 0, 0
        if hasattr(super(), "before_training_epoch"):
            super().before_training_epoch()


    def training_epoch_end(self):
        if self.debug:
            avg_ = lambda cor_count: cor_count[0]/cor_count[1]
            classes = self.class_acc.keys()
            classes_s0 = self.class_acc_s0.keys()
            classes_s1 = self.class_acc_s1.keys()
            target_classes = list(set(classes).intersection(classes_s0).intersection(classes_s1))
            if len(classes)!=len(target_classes):
                print(f"{classes=}, {target_classes=}")
            if self.params['metric'] == "EO":
                disp = [abs(avg_(self.class_acc_s0[c]) - avg_(self.class_acc_s1[c])) for c in target_classes]
                CELoss_disp = [abs(avg_(self.class_loss_s0[c]) - avg_(self.class_loss_s1[c])) for c in target_classes]
            elif self.params['metric'] == "DP":
                # disp = [0.5*(abs(self.class_pred_count_s0[c]/self.count_s0 - self.class_pred_count[c]/self.count) + \
                #              abs(self.class_pred_count_s1[c]/self.count_s1 - self.class_pred_count[c]/self.count)) for c in target_classes]
                # CELoss_disp = [0.5*abs(self.class_pred_count_s0[c]/self.count_s0*avg_(self.class_loss_s0[c]) - \
                #                        self.class_pred_count_s1[c]/self.count_s1)*avg_(self.class_loss_s1[c]) for c in target_classes]
                disp = [0.5*(abs(self.class_pred_count_s0.get(c, 0)/self.count_s0 - self.class_pred_count[c]/self.count) + \
                             abs(self.class_pred_count_s1.get(c, 0)/self.count_s1 - self.class_pred_count[c]/self.count)) for c in target_classes]
                CELoss_disp = [0.5*(abs(self.class_pred_count_s0.get(c, 0)/self.count_s0*avg_(self.class_loss_s0[c]) - self.class_pred_count[c]/self.count*avg_(self.class_loss[c])) + \
                             abs(self.class_pred_count_s1.get(c, 0)/self.count_s1*avg_(self.class_loss_s0[c]) - self.class_pred_count[c]/self.count*avg_(self.class_loss[c]))) for c in target_classes]
            else:
                raise NotImplementedError

            self.disp.append(disp)
            self.CELoss_disp.append(CELoss_disp)
        return super().training_epoch_end()

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
        grad_data, grad_data2, grads = None, None, None

        loaded_batch = list()
        self.backbone.eval()
        self.backbone.zero_grad()

        # in case of different fairness loss
        fairloss_type = self.params.get("fairness_loss", "CE")

        if fairloss_type == "CE":
            pass
        elif fairloss_type == "hinge":
            classwise_loss_s0_2 = copy.deepcopy(classwise_loss_s0)
            classwise_grad_s0_2 = copy.deepcopy(classwise_grad_s0)
            classwise_loss_s1_2 = copy.deepcopy(classwise_loss_s1)
            classwise_grad_s1_2 = copy.deepcopy(classwise_grad_s1)
        else:
            raise NotImplementedError

        for batch_idx, items in enumerate(loader):
            # print(f"{batch_idx=}")
            inp, targ, t_id, indices, sample_weight, sensitive_label, *_ = items
            if current_set:
                loaded_batch.append(items)
                pass
            # self.backbone.forward
            inp, targ, t_id, sensitive_label  = inp, targ.to(device), t_id.to(device), sensitive_label.to(device)
            if isinstance(inp, list):
                inp = [x.to(device) for x in inp]
            else:
                inp = inp.to(device)

            pred, embeds = self.backbone.forward_embeds(inp, t_id)
            self.pred_shape = pred.shape[1]
            self.embeds_shape = embeds.shape[1]
            criterion.reduction = "none"
            loss = criterion(pred, targ.reshape(-1))
            criterion.reduction = "mean"

            # ADDED
            if fairloss_type == "CE":
                loss2 = loss
            elif fairloss_type == "hinge":
                loss2 = hinge(pred, targ.reshape(-1))
            else:
                raise NotImplementedError

            bias_grads = torch.autograd.grad(loss.mean(), pred)[0]
            bias_expand = torch.repeat_interleave(bias_grads, embeds.shape[1], dim=1)
            weight_grads = bias_expand * embeds.repeat(1, pred.shape[1])
            grads = torch.cat([bias_grads.clone().detach().cpu(), weight_grads.clone().detach().cpu()], dim=1)
            grads = grads.cpu()

            if fairloss_type != "CE":
                bias_grads2 = torch.autograd.grad(loss2.mean(), pred)[0]
                bias_expand2 = torch.repeat_interleave(bias_grads2, embeds.shape[1], dim=1)
                weight_grads2 = bias_expand2 * embeds.repeat(1, pred.shape[1])
                grads2 = torch.cat([bias_grads2.clone().detach().cpu(), weight_grads2.clone().detach().cpu()], dim=1)
                grads2 = grads2.cpu()

                del bias_grads2, weight_grads2

            del bias_grads, weight_grads

            if self.params['dataset'] == "BiasedMNIST":
                sensitive = torch.ne(targ, sensitive_label)
            else:
                sensitive = sensitive_label
            sensitive = sensitive.long()

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
                
                if fairloss_type != "CE":
                    if sensitive[i].cpu().item() == 0:
                        classwise_loss_s0_2[e.cpu().item()].append(loss2[i].detach().cpu())
                        classwise_grad_s0_2[e.cpu().item()].append(grads2[i].detach().cpu())
                    elif sensitive[i].cpu().item() == 1:
                        classwise_loss_s1_2[e.cpu().item()].append(loss2[i].detach().cpu())
                        classwise_grad_s1_2[e.cpu().item()].append(grads2[i].detach().cpu())
            
            if isinstance(inp, list):
                del inp[:]
            del inp, targ, t_id, sensitive_label

            grad_data = grads if grad_data is None else torch.cat([grad_data, grads])

            self.backbone.zero_grad()

        classwise_loss_all = [classwise_loss_s0, classwise_loss_s1]
        classwise_grad_all = [classwise_grad_s0, classwise_grad_s1]

        if fairloss_type != "CE":
            classwise_loss_all_2 = [classwise_loss_s0_2, classwise_loss_s1_2]
            classwise_grad_all_2 = [classwise_grad_s0_2, classwise_grad_s1_2]

            classwise_loss_all = (classwise_loss_all, classwise_loss_all_2)
            classwise_grad_all = (classwise_grad_all, classwise_grad_all_2)

        return classwise_loss_all, classwise_grad_all, grad_data, loaded_batch

    def get_loss_grad_all(self, task_id): # type: ignore
        num_workers = self.params.get('num_dataloader_workers', torch.get_num_threads())

        classwise_loss_all, classwise_grad_all, grad_data_prev, *_ = self.get_loss_grad(task_id, self.episodic_memory_loader, current_set = False)
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=False,
                                   num_workers=num_workers, pin_memory=True)[0]
        current_loss_all, current_grad_all, grad_data_current, new_batch = self.get_loss_grad(task_id, train_loader, current_set = True)
        grad_data_current = grad_data_current[self.non_select_indexes] # type: ignore

        fairloss_type = self.params.get("fairness_loss", "CE")
        if fairloss_type != "CE":
            classwise_loss_all, classwise_loss_all_2 = classwise_loss_all
            classwise_grad_all, classwise_grad_all_2 = classwise_grad_all

            current_loss_all, current_loss_all_2 = current_loss_all
            current_grad_all, current_grad_all_2 = current_grad_all

            classwise_loss_all_2[0].update(current_loss_all_2[0])
            classwise_loss_all_2[1].update(current_loss_all_2[1])
            classwise_grad_all_2[0].update(current_grad_all_2[0])
            classwise_grad_all_2[1].update(current_grad_all_2[1])

        classwise_loss_all[0].update(current_loss_all[0])
        classwise_loss_all[1].update(current_loss_all[1])
        classwise_grad_all[0].update(current_grad_all[0])
        classwise_grad_all[1].update(current_grad_all[1])

        loss_group = []
        grads = []
        for i, classwise_loss in enumerate(classwise_loss_all):
            for k, v in classwise_loss.items():
                v3 = classwise_grad_all[i][k]
                if len(v) == 0 :
                    print(f"###classwise_loss of {k}, s={i} is missing###")
                    raise NotImplementedError
                loss_ = torch.stack(v).mean(dim=0).view(1, -1)
                grads_ = torch.stack(v3).mean(dim=0).view(1, -1)
                loss_group.append(loss_)
                grads.append(grads_)

        if fairloss_type != "CE":
            loss_group2 = []
            grads2 = []
            for i, classwise_loss2 in enumerate(classwise_loss_all_2):
                for k, v in classwise_loss2.items():
                    v3 = classwise_grad_all_2[i][k]
                    if len(v) == 0 :
                        print(f"###classwise_loss of {k}, s={i} is missing###")
                        raise NotImplementedError
                    loss_ = torch.stack(v).mean(dim=0).view(1, -1)
                    grads_ = torch.stack(v3).mean(dim=0).view(1, -1)

                    loss_group2.append(loss_)
                    grads2.append(grads_)

        with torch.no_grad():
            loss_group = torch.cat(loss_group, dim=0).view(1,-1)
            grad_group = torch.cat(grads, dim=0)
            self.classwise_mean_grad.append(torch.norm(grad_group, dim=1))

            # normalize to make class/group difference be similar
            if self.params.get('no_class_grad_norm', False):
                grad_group = grad_group
            else:
                grad_group = F.normalize(grad_group, p=2, dim=1) # (num_class) * (weight&bias dim)
            if self.params.get('no_datapoints_grad_norm', False):
                grad_data_prev = grad_data_prev
                grad_data_current = grad_data_current
            else:
                grad_data_prev = F.normalize(grad_data_prev, p=2, dim=1) # (num_training) * (weight&bias dim)
                grad_data_current = F.normalize(grad_data_current, p=2, dim=1) # (num_training) * (weight&bias dim)

            if fairloss_type != "CE":
                loss_group2 = torch.cat(loss_group2, dim=0).view(1,-1)
                grad_group2 = torch.cat(grads2, dim=0)
                grad_group2 = F.normalize(grad_group2, p=2, dim=1) # (num_class) * (weight&bias dim)

        if fairloss_type != "CE":
            loss_group = (loss_group, loss_group2)
            grad_group = (grad_group, grad_group2)


        return loss_group, grad_group, (grad_data_prev, grad_data_current), new_batch
    

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
        grad_data, grads = None, None

        for batch_idx, items in enumerate(loader):
            inp, targ, t_id, indices, sample_weight, sensitive_label, *_ = items
            # self.backbone.forward
            item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
            inp, targ, t_id, indices, sample_weight, sensitive_label, *_ = item_to_devices
            if isinstance(inp, list):
                inp = [x.to(device) for x in inp]
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

            model.zero_grad()

        classwise_loss_all = [classwise_loss_s0, classwise_loss_s1]
        classwise_grad_all = [classwise_grad_s0, classwise_grad_s1]
        return classwise_loss_all, classwise_grad_all


    def measure_loss(self, task_id, model):
        num_workers = self.params.get('num_dataloader_workers', torch.get_num_threads())

        classwise_loss_all, classwise_grad_all, *_ = self.get_loss_grad_model(task_id, self.episodic_memory_loader, model=model, current_set = False)
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=False,
                                   num_workers=num_workers, pin_memory=True)[0]
        current_loss_all, current_grad_all, *_ = self.get_loss_grad_model(task_id, train_loader, model=model, current_set = True)

        classwise_loss_all[0].update(current_loss_all[0])
        classwise_loss_all[1].update(current_loss_all[1])
        classwise_grad_all[0].update(current_grad_all[0])
        classwise_grad_all[1].update(current_grad_all[1])

        loss_group = []
        grads = []
        for i, classwise_loss in enumerate(classwise_loss_all):
            for k, v in classwise_loss.items():
                v3 = classwise_grad_all[i][k]
                if len(v) == 0 :
                    print(f"###classwise_loss of {k}, s={i} is missing###")
                    raise NotImplementedError

                loss_ = torch.stack(v).mean(dim=0).view(1, -1)
                grads_ = torch.stack(v3).mean(dim=0).view(1, -1)
                loss_group.append(loss_)
                grads.append(grads_)

        with torch.no_grad():
            loss_group = torch.cat(loss_group, dim=0).view(1,-1)
        return loss_group


    def converter_LP_absolute_additional_EO(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        """
        LP with linear additional term
        input: 
            loss_group: torch.cat([prev_classwise_loss_s0, prev_classwise_loss_s1])
            alpha: coefficient for gradients
            grad_group: torch.cat([prev_classwise_gradient_s0, prev_classwise_gradient_s1])
            grad_data: current data pointwise gradient
        output: 
            A: averaged alpha*1/2*(prev_classwise_gradient_s0 - prev_classwise_gradient_s1)·pointwise_gradient
            b: averaged 1/2*(prev_classwise_loss_s0 - prev_classwise_loss_s1)
            C: averaged alpha*classwise_gradient·pointwise_gradient
            d: averaged prev_classwise_loss
        where A, b are linear coefficient for absolute term, C, d are non-absolute term
        min_x |b - Ax| + (d - Cx)
        """
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        loss_group = torch.transpose(loss_group, 0, 1) # (num_class) * (1)
        grad_group = torch.transpose(grad_group, 0, 1) # (num_class) * (weight&bias dim)
        
        n = len(loss_group)//2
        # m, dim = grad_data.shape

        b = torch.zeros([n, 1], device=device)
        grad_diff = torch.zeros([grad_group.shape[0], n], device=device)

        d = torch.zeros([n, 1], device=device)
        classwise_grads = torch.zeros([grad_group.shape[0], n], device=device)

        for j in range(n):
            b[j] = (loss_group[j] - loss_group[n+j])/2  # |Z|=2
            grad_diff[:,j] = (grad_group[:,j] - grad_group[:,n+j])/2  # |Z|=2

            d[j] = (loss_group[j] + loss_group[n+j])/2
            classwise_grads[:,j] = (grad_group[:,j] + grad_group[:,n+j])/2

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

    def converter_LP_absolute_additional_EO_diffloss(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        """
        LP with linear additional term
        input: 
            loss_group: torch.cat([prev_classwise_loss_s0, prev_classwise_loss_s1])
            alpha: coefficient for gradients
            grad_group: torch.cat([prev_classwise_gradient_s0, prev_classwise_gradient_s1])
            grad_data: current data pointwise gradient
        output: 
            A: averaged alpha*1/2*(prev_classwise_gradient_s0 - prev_classwise_gradient_s1)·pointwise_gradient
            b: averaged 1/2*(prev_classwise_loss_s0 - prev_classwise_loss_s1)
            C: averaged alpha*classwise_gradient·pointwise_gradient
            d: averaged prev_classwise_loss
        where A, b are linear coefficient for absolute term, C, d are non-absolute term
        min_x |b - Ax| + (d - Cx)
        """
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        fairloss_type = self.params.get("fairness_loss", "CE")
        if fairloss_type == "CE":
            raise ValueError

        loss_group, loss_group2 = loss_group
        grad_group, grad_group2 = grad_group

        loss_group = torch.transpose(loss_group, 0, 1) # (num_class) * (1)
        grad_group = torch.transpose(grad_group, 0, 1) # (num_class) * (weight&bias dim)

        loss_group2 = torch.transpose(loss_group2, 0, 1)
        grad_group2 = torch.transpose(grad_group2, 0, 1) # (weight&bias 차원수) * (num_class)

        n = len(loss_group)//2
        # m, dim = grad_data.shape

        b = torch.zeros([n, 1], device=device)
        grad_diff = torch.zeros([grad_group.shape[0], n], device=device)
        grad_diff2 = torch.zeros([grad_group2.shape[0], n], device=device)

        d = torch.zeros([n, 1], device=device)
        classwise_grads = torch.zeros([grad_group.shape[0], n], device=device)

        for j in range(n):
            b[j] = (loss_group2[j] - loss_group2[n+j])/2  # |Z|=2
            grad_diff2[:,j] = (grad_group2[:,j] - grad_group2[:,n+j])/2  # |Z|=2

            d[j] = (loss_group[j] + loss_group[n+j])/2
            classwise_grads[:,j] = (grad_group[:,j] + grad_group[:,n+j])/2

        grad_diff2 = alpha*grad_diff2
        A = torch.matmul(grad_diff2.T, grad_data.T)

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


    def converter_LP_absolute_additional_DP(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        """
        LP with linear additional term
        input: 
            loss_group: torch.cat([prev_classwise_loss_s0, prev_classwise_loss_s1])
            alpha: coefficient for gradients
            grad_group: torch.cat([prev_classwise_gradient_s0, prev_classwise_gradient_s1])
            grad_data: current data pointwise gradient
        output: 
            A: averaged alpha*1/2*(prev_classwise_gradient_s0 - prev_classwise_gradient_s1)·pointwise_gradient
            b: averaged 1/2*(prev_classwise_loss_s0 - prev_classwise_loss_s1)
            C: averaged alpha*classwise_gradient·pointwise_gradient
            d: averaged prev_classwise_loss
        where A, b are linear coefficient for absolute term, C, d are non-absolute term
        min_x |b - Ax| + (d - Cx)
        """
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        loss_group = torch.transpose(loss_group, 0, 1) # (num_class) * (1)
        grad_group = torch.transpose(grad_group, 0, 1) # (num_class) * (weight&bias dim)
        
        n = len(loss_group)//2
        def m_y_z(y, z):
            return self.benchmark.m_dict[z][y]
        def m_z(z, y_all=n):
            return np.sum([m_y_z(y, z) for y in self.benchmark.class_idx[:y_all]])

        # m, dim = grad_data.shape

        b = torch.zeros([n, 1], device=device)
        grad_diff = torch.zeros([grad_group.shape[0], n], device=device)

        d = torch.zeros([n, 1], device=device)
        classwise_grads = torch.zeros([grad_group.shape[0], n], device=device)

        for j in range(n):
            b[j] = (m_y_z(j, 0) / m_z(0) * loss_group[j] - m_y_z(j, 1) / m_z(1) * loss_group[n+j])/2 # |Z|=2
            grad_diff[:,j] = (m_y_z(j, 0) / m_z(0) * grad_group[:,j] - m_y_z(j, 1) / m_z(1) * grad_group[:,n+j])/2

            d[j] = (loss_group[j] + loss_group[n+j])/2
            classwise_grads[:,j] = (grad_group[:,j] + grad_group[:,n+j])/2

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
    
    def converter_LP_absolute_additional_DP_diffloss(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        """
        LP with linear additional term
        input: 
            loss_group: torch.cat([prev_classwise_loss_s0, prev_classwise_loss_s1])
            alpha: coefficient for gradients
            grad_group: torch.cat([prev_classwise_gradient_s0, prev_classwise_gradient_s1])
            grad_data: current data pointwise gradient
        output: 
            A: averaged alpha*1/2*(prev_classwise_gradient_s0 - prev_classwise_gradient_s1)·pointwise_gradient
            b: averaged 1/2*(prev_classwise_loss_s0 - prev_classwise_loss_s1)
            C: averaged alpha*classwise_gradient·pointwise_gradient
            d: averaged prev_classwise_loss
        where A, b are linear coefficient for absolute term, C, d are non-absolute term
        min_x |b - Ax| + (d - Cx)
        """
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        fairloss_type = self.params.get("fairness_loss", "CE")
        if fairloss_type == "CE":
            raise ValueError
        
        loss_group, loss_group2 = loss_group
        grad_group, grad_group2 = grad_group

        loss_group = torch.transpose(loss_group, 0, 1) # (num_class) * (1)
        grad_group = torch.transpose(grad_group, 0, 1) # (num_class) * (weight&bias dim)

        loss_group2 = torch.transpose(loss_group2, 0, 1)
        grad_group2 = torch.transpose(grad_group2, 0, 1) # (weight&bias 차원수) * (num_class)

        n = len(loss_group)//2
        def m_y_z(y, z):
            return self.benchmark.m_dict[z][y]
        def m_z(z, y_all=n):
            return np.sum([m_y_z(y, z) for y in self.benchmark.class_idx[:y_all]])

        # m, dim = grad_data.shape

        b = torch.zeros([n, 1], device=device)
        grad_diff = torch.zeros([grad_group.shape[0], n], device=device)
        grad_diff2 = torch.zeros([grad_group2.shape[0], n], device=device)

        d = torch.zeros([n, 1], device=device)
        classwise_grads = torch.zeros([grad_group.shape[0], n], device=device)

        for j in range(n):
            b[j] = (m_y_z(j, 0) / m_z(0) * loss_group2[j] - m_y_z(j, 1) / m_z(1) * loss_group2[n+j])/2 # |Z|=2
            grad_diff[:,j] = (m_y_z(j, 0) / m_z(0) * grad_group2[:,j] - m_y_z(j, 1) / m_z(1) * grad_group2[:,n+j])/2

            d[j] = (loss_group[j] + loss_group[n+j])/2
            classwise_grads[:,j] = (grad_group[:,j] + grad_group[:,n+j])/2

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


    def converter_LP_absolute_additional_EO_v1(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        """
        LP with linear additional term
        input: 
            loss_group: torch.cat([prev_classwise_loss_s0, prev_classwise_loss_s1])
            alpha: coefficient for gradients
            grad_group: torch.cat([prev_classwise_gradient_s0, prev_classwise_gradient_s1])
            grad_data: current data pointwise gradient
        output: 
            A: averaged alpha*1/2*(prev_classwise_gradient_s0 - prev_classwise_gradient_s1)·pointwise_gradient
            b: averaged 1/2*(prev_classwise_loss_s0 - prev_classwise_loss_s1)
                - averaged tau*alpha*1/2*(prev_classwise_gradient_s0 - prev_classwise_gradient_s1)·buffer_pointwise_gradient
            C: averaged alpha*classwise_gradient·pointwise_gradient
            d: averaged prev_classwise_loss
                - averaged tau*alpha*classwise_gradient·buffer_pointwise_gradient

        where A, b are linear coefficient for absolute term, C, d are non-absolute term
        min_x |b - Ax| + (d - Cx)
        """
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        loss_group = torch.transpose(loss_group, 0, 1) # (num_class) * (1)
        grad_group = torch.transpose(grad_group, 0, 1) # (num_class) * (weight&bias dim)
        grad_data_prev = kwargs['grad_data_prev']
        tau = self.params['tau']

        n = len(loss_group)//2
        # m, dim = grad_data.shape

        b = torch.zeros([n, 1], device=device)
        grad_diff = torch.zeros([grad_group.shape[0], n], device=device)

        d = torch.zeros([n, 1], device=device)
        classwise_grads = torch.zeros([grad_group.shape[0], n], device=device)

        for j in range(n):
            b[j] = (loss_group[j] - loss_group[n+j])/2  # |Z|=2
            grad_diff[:,j] = (grad_group[:,j] - grad_group[:,n+j])/2  # |Z|=2

            d[j] = (loss_group[j] + loss_group[n+j])/2
            classwise_grads[:,j] = (grad_group[:,j] + grad_group[:,n+j])/2

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

    def converter_LP_absolute_additional_DP_v1(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        """
        LP with linear additional term
        input: 
            loss_group: torch.cat([prev_classwise_loss_s0, prev_classwise_loss_s1])
            alpha: coefficient for gradients
            grad_group: torch.cat([prev_classwise_gradient_s0, prev_classwise_gradient_s1])
            grad_data: current data pointwise gradient
        output: 
            A: averaged alpha*1/2*(prev_classwise_gradient_s0 - prev_classwise_gradient_s1)·pointwise_gradient
            b: averaged 1/2*(prev_classwise_loss_s0 - prev_classwise_loss_s1)
                - averaged tau*alpha*1/2*(prev_classwise_gradient_s0 - prev_classwise_gradient_s1)·buffer_pointwise_gradient
            C: averaged alpha*classwise_gradient·pointwise_gradient
            d: averaged prev_classwise_loss
                - averaged tau*alpha*classwise_gradient·buffer_pointwise_gradient
        where A, b are linear coefficient for absolute term, C, d are non-absolute term
        min_x |b - Ax| + (d - Cx)
        """
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        loss_group = torch.transpose(loss_group, 0, 1) # (num_class) * (1)
        grad_group = torch.transpose(grad_group, 0, 1) # (num_class) * (weight&bias dim)
        grad_data_prev = kwargs['grad_data_prev']
        tau = self.params['tau']
        
        n = len(loss_group)//2
        def m_y_z(y, z):
            return self.benchmark.m_dict[z][y]
        def m_z(z, y_all=n):
            return np.sum([m_y_z(y, z) for y in self.benchmark.class_idx[:y_all]])

        # m, dim = grad_data.shape

        b = torch.zeros([n, 1], device=device)
        grad_diff = torch.zeros([grad_group.shape[0], n], device=device)

        d = torch.zeros([n, 1], device=device)
        classwise_grads = torch.zeros([grad_group.shape[0], n], device=device)

        for j in range(n):
            b[j] = (m_y_z(j, 0) / m_z(0) * loss_group[j] - m_y_z(j, 1) / m_z(1) * loss_group[n+j])/2 # |Z|=2
            grad_diff[:,j] = (m_y_z(j, 0) / m_z(0) * grad_group[:,j] - m_y_z(j, 1) / m_z(1) * grad_group[:,n+j])/2

            d[j] = (loss_group[j] + loss_group[n+j])/2
            classwise_grads[:,j] = (grad_group[:,j] + grad_group[:,n+j])/2

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


    def converter_LP_absolute_additional_EO_v2(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        """
        For (absolute) loss difference calculation, only measure loss for past classes
        min_x |b - Ax| + (d - Cx)
        where A, b contain only pass classes
        """
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        loss_group = torch.transpose(loss_group, 0, 1) # (num_class) * (1)
        grad_group = torch.transpose(grad_group, 0, 1) # (num_class) * (weight&bias dim)
        
        n = len(loss_group)//2
        num_past_classes = n-num_current_classes
        # m, dim = grad_data.shape

        b = torch.zeros([n, 1], device=device)
        grad_diff = torch.zeros([grad_group.shape[0], n], device=device)

        d = torch.zeros([n, 1], device=device)
        classwise_grads = torch.zeros([grad_group.shape[0], n], device=device)

        for j in range(n):
            if not j >= num_past_classes:
                b[j] = (loss_group[j] - loss_group[n+j])/2  # |Z|=2
                grad_diff[:,j] = (grad_group[:,j] - grad_group[:,n+j])/2  # |Z|=2

            d[j] = (loss_group[j] + loss_group[n+j])/2
            classwise_grads[:,j] = (grad_group[:,j] + grad_group[:,n+j])/2

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

    def converter_LP_absolute_additional_DP_v2(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        """
        LP with linear additional term
        input: 
            loss_group: torch.cat([prev_classwise_loss_s0, prev_classwise_loss_s1])
            alpha: coefficient for gradients
            grad_group: torch.cat([prev_classwise_gradient_s0, prev_classwise_gradient_s1])
            grad_data: current data pointwise gradient
        output: 
            A: averaged alpha*1/2*(prev_classwise_gradient_s0 - prev_classwise_gradient_s1)·pointwise_gradient
            b: averaged 1/2*(prev_classwise_loss_s0 - prev_classwise_loss_s1)
            C: averaged alpha*classwise_gradient·pointwise_gradient
            d: averaged prev_classwise_loss
        where A, b are linear coefficient for absolute term, C, d are non-absolute term
        min_x |b - Ax| + (d - Cx)
        """
        device = 'cpu'
        num_current_classes = self.get_num_current_classes(task)
        loss_group = torch.transpose(loss_group, 0, 1) # (num_class) * (1)
        grad_group = torch.transpose(grad_group, 0, 1) # (num_class) * (weight&bias dim)
        
        n = len(loss_group)//2
        num_past_classes = n-num_current_classes

        def m_y_z(y, z):
            return self.benchmark.m_dict[z][y]
        def m_z(z, y_all=num_past_classes):
            return np.sum([m_y_z(y, z) for y in self.benchmark.class_idx[:y_all]])

        # m, dim = grad_data.shape

        b = torch.zeros([n, 1], device=device)
        grad_diff = torch.zeros([grad_group.shape[0], n], device=device)

        d = torch.zeros([n, 1], device=device)
        classwise_grads = torch.zeros([grad_group.shape[0], n], device=device)
        
        for j in range(n):
            if not j >= num_past_classes:
                b[j] = (m_y_z(j, 0) / m_z(0) * loss_group[j] - m_y_z(j, 1) / m_z(1) * loss_group[n+j])/2 # |Z|=2
                grad_diff[:,j] = (m_y_z(j, 0) / m_z(0) * grad_group[:,j] - m_y_z(j, 1) / m_z(1) * grad_group[:,n+j])/2

            d[j] = (loss_group[j] + loss_group[n+j])/2
            classwise_grads[:,j] = (grad_group[:,j] + grad_group[:,n+j])/2

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

    def converter_LP_no_losses(self, loss_group, alpha, grad_group, grad_data, task=None, **kwargs):
        device = 'cpu'
        loss_group = torch.transpose(loss_group, 0, 1)
        grad_group = torch.transpose(grad_group, 0, 1) # (weight&bias dim) * (num_class)
        
        n = len(loss_group)//2
        # m, dim = grad_data.shape

        b = torch.zeros([n, 1], device=device)
        grad_diff = torch.zeros([grad_group.shape[0], n], device=device)

        for j in range(n):
            b[j] = loss_group[j] - loss_group[n+j]
            grad_diff[:,j] = grad_group[:,j] - grad_group[:,n+j]

        grad_diff = alpha*grad_diff
        A = torch.matmul(grad_diff.T, grad_data.T)

        A, b = A/n, b/n
        return A, b

    def prepare_train_loader(self, task_id, epoch=0):
        solver = self.params.get('solver')
        metric = self.params.get('metric')
        agg = self.params.get('fairness_agg')
        fairloss_type = self.params.get("fairness_loss", "CE")

        if solver is None:
            if metric == "EO" or metric is None:
                if fairloss_type == "CE":
                    if agg == "mean" or agg is None:
                        solver = absolute_and_nonabsolute_minsum_LP_solver
                        self.converter = self.converter_LP_absolute_additional_EO
                    elif agg == "max":
                        raise NotImplementedError
                        solver = absolute_minimax_LP_solver
                        self.converter = self.converter_LP_absolute_only_EO
                    else:
                        raise NotImplementedError
                else:
                    solver = absolute_and_nonabsolute_minsum_LP_solver
                    self.converter = self.converter_LP_absolute_additional_EO_diffloss
            elif metric == "DP":
                if fairloss_type == "CE":
                    if agg == "mean" or agg is None:
                        solver = absolute_and_nonabsolute_minsum_LP_solver
                        self.converter = self.converter_LP_absolute_additional_DP
                    else:
                        raise NotImplementedError
                else:
                    solver = absolute_and_nonabsolute_minsum_LP_solver
                    self.converter = self.converter_LP_absolute_additional_DP_diffloss

            elif metric == "no_metrics":
                pass
            else:
                raise NotImplementedError
        else:
            self.converter = self.params.get('converter')
        return super().prepare_train_loader(task_id, solver=solver, epoch=epoch)

    def training_step(self, task_ids, inp, targ, optimizer, criterion, sample_weight=None, sensitive_label=None):
        optimizer.zero_grad()
        prob = self.backbone(inp, task_ids)
        pred = prob.data.max(1, keepdim=True)[1]
        criterion.reduction = "none"
        loss = criterion(prob, targ)
        criterion.reduction = "mean"
        if sample_weight is not None:
            loss = loss*sample_weight
            loss_raw = loss.detach().clone()
        loss = loss.mean()
        loss.backward()
        if (task_ids[0] > 1) and self.params['tau']:
            grad_batch = flatten_grads(self.backbone).detach().clone()
            optimizer.zero_grad()

            # get grad_ref
            inp_ref, targ_ref, task_ids_ref, *_ = self.sample_batch_from_memory()
            prob_ref = self.backbone(inp_ref, task_ids_ref)
            pred_ref = prob_ref.data.max(1, keepdim=True)[1]
            criterion.reduction = "none"
            loss = criterion(prob_ref, targ_ref.reshape(len(targ_ref)))
            loss_ref_raw = loss.detach().clone()
            criterion.reduction = "mean"
            loss = loss.mean()
            loss.backward()
            grad_ref = flatten_grads(self.backbone).detach().clone()
            grad_batch += self.params['tau']*grad_ref

            optimizer.zero_grad()
            self.backbone = assign_grads(self.backbone, grad_batch)
        optimizer.step()

        ## For debugging
        if self.debug:
            same = pred.eq(targ.data.view_as(pred))

            if self.benchmark.__class__.__name__ == "BiasedMNIST":
                sensitive_label = process_for_biasedmnist(sensitive_label, targ)

            for p, t, s, l, sen in zip(pred, targ, same, loss_raw, sensitive_label):
                p = p.cpu().item()
                t = t.cpu().item()
                s = s.cpu().item()
                l = l.cpu().item()
                sen = sen.cpu().item()
                self.class_acc[t] = self.class_acc.get(t, np.array([0, 0])) + np.array([s, 1])
                self.class_loss[t] = self.class_loss.get(t, np.array([0, 0])) + np.array([l, 1])
                self.class_pred_count[p] = self.class_pred_count.get(p, 0) + 1
                self.count+=1
                if sen == 0:
                    self.class_acc_s0[t] = self.class_acc_s0.get(t, np.array([0, 0])) + np.array([s, 1])
                    self.class_loss_s0[t] = self.class_loss_s0.get(t, np.array([0, 0])) + np.array([l, 1])
                    self.class_pred_count_s0[p] = self.class_pred_count_s0.get(p, 0) + 1
                    self.count_s0+=1
                elif sen == 1:
                    self.class_acc_s1[t] = self.class_acc_s1.get(t, np.array([0, 0])) + np.array([s, 1])
                    self.class_loss_s1[t] = self.class_loss_s1.get(t, np.array([0, 0])) + np.array([l, 1])
                    self.class_pred_count_s1[p] = self.class_pred_count_s1.get(p, 0) + 1
                    self.count_s1+=1
                else:
                    raise NotImplementedError
        
            if (task_ids[0] > 1) and self.params['tau']:
                same_ref = pred_ref.eq(targ_ref.data.view_as(pred_ref))
                for p, t, s, l, sen in zip(pred_ref, targ_ref, same_ref, loss_ref_raw, sensitive_label):
                    p = p.cpu().item()
                    t = t.cpu().item()
                    s = s.cpu().item()
                    l = l.cpu().item()
                    sen = sen.cpu().item()
                    self.class_acc[t] = self.class_acc.get(t, np.array([0, 0])) + np.array([s, 1])
                    self.class_loss[t] = self.class_loss.get(t, np.array([0, 0])) + np.array([l, 1])
                    self.class_pred_count[p] = self.class_pred_count.get(p, 0) + 1
                    self.count+=1
                    if sen == 0:
                        self.class_acc_s0[t] = self.class_acc_s0.get(t, np.array([0, 0])) + np.array([s, 1])
                        self.class_loss_s0[t] = self.class_loss_s0.get(t, np.array([0, 0])) + np.array([l, 1])
                        self.class_pred_count_s0[p] = self.class_pred_count_s0.get(p, 0) + 1
                        self.count_s0+=1
                    elif sen == 1:
                        self.class_acc_s1[t] = self.class_acc_s1.get(t, np.array([0, 0])) + np.array([s, 1])
                        self.class_loss_s1[t] = self.class_loss_s1.get(t, np.array([0, 0])) + np.array([l, 1])
                        self.class_pred_count_s1[p] = self.class_pred_count_s1.get(p, 0) + 1
                        self.count_s1+=1
                    else:
                        raise NotImplementedError

def process_for_biasedmnist(sensitive_label, targ):
    sensitive = torch.ne(targ, sensitive_label)
    sensitive = sensitive.long()
    return sensitive

