import torch
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.utils import flatten_grads, assign_grads
from torch.nn.functional import relu, avg_pool2d
import matplotlib.pyplot as plt

import copy
import os

from .imbalance import Heuristic2

class Heuristic1(Heuristic2):
    def get_loss_grad(self, task_id, loader, current_set = False):
        criterion = self.prepare_criterion(task_id)
        device = self.params['device']
        inc_num = 2 # MNIST
        if current_set:
            classwise_loss = {x:list() for x in self.benchmark.class_idx[(task_id-1)*inc_num:task_id*inc_num]}
            classwise_grad_dict = {x:list() for x in self.benchmark.class_idx[(task_id-1)*inc_num:task_id*inc_num]}
        else:
            classwise_loss = {x:list() for x in self.benchmark.class_idx[:(task_id-1)*inc_num]}
            classwise_grad_dict = {x:list() for x in self.benchmark.class_idx[:(task_id-1)*inc_num]}
        new_grads, grads = None, None
        
        new_num = 0
        for batch_idx, (inp, targ, t_id, *_) in enumerate(loader):
            # self.backbone.forward
            inp, targ, t_id  = inp.to(device), targ.to(device), t_id.to(device)
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

            for i, e in enumerate(targ):
                # new_num+=1
                classwise_loss[e.cpu().item()].append(loss[i])
                classwise_grad_dict[e.cpu().item()].append(grads[i])

            if current_set:
                new_grads = grads if new_grads is None else torch.cat([new_grads, grads])

            self.backbone.zero_grad()

        # for x in self.benchmark.class_idx:
        #     if len(classwise_loss[x]) == 0:
        #         del classwise_loss[x]
        #         del classwise_grad_dict[x]
        # print(f"{new_num=}")
        if current_set:
            for k in classwise_loss:
                if len(classwise_loss[k]) == 0:
                    print(f"{k} is missing")
                    for kk in classwise_loss:
                        print(f"{len(classwise_loss[kk])=}")
                    # classwise_loss[k].append(torch.empty((0, self.pred_shape*(self.embeds_shape+1)), device=device, dtype=torch.float32))
                    # classwise_loss[k].append(torch.empty((0, self.pred_shape*(self.embeds_shape+1)), device=device, dtype=torch.float32))
                    classwise_loss[k].append(torch.zeros((0, self.pred_shape*(self.embeds_shape+1)), device=device, dtype=torch.float32))
                    classwise_grad_dict[k].append(torch.zeros((0, self.pred_shape*(self.embeds_shape+1)), device=device, dtype=torch.float32))

            # new_grads = torch.empty((0, pred.shape[1]*(embeds.shape[1]+1)), device=device, dtype=torch.float32)
            
            # new_grads = torch.empty((0, self.pred_shape*(self.embeds_shape+1)), device=device, dtype=torch.float32)

        return classwise_loss, classwise_grad_dict, new_grads
    
    def get_loss_grad_all(self, task_id):
        num_workers = self.params.get('num_dataloader_workers', 0)

        classwise_loss, classwise_grad_dict, _ = self.get_loss_grad(task_id, self.episodic_memory_loader, current_set = False)
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=False,
                                   num_workers=num_workers, pin_memory=True)[0]
        current_loss, current_grad_dict, new_grads = self.get_loss_grad(task_id, train_loader, current_set = True)
        r_new_grads = new_grads[self.non_select_indexes]
        classwise_loss.update(current_loss)
        classwise_grad_dict.update(current_grad_dict)

        losses = []
        grads = []
        for k, v in classwise_loss.items():
            v3 = classwise_grad_dict[k]
            loss_ = torch.stack(v).mean(dim=0).view(1, -1).detach().clone()
            grads_ = torch.stack(v3).mean(dim=0).view(1, -1).detach().clone()
            # if loss_.shape[1] == 0:
            #     loss_ = torch.zeros([1, 1]).to(self.params['device'])
            #     grads_ = torch.zeros([1, self.pred_shape*(self.embeds_shape+1)]).to(self.params['device'])
            losses.append(loss_)
            grads.append(grads_)
            # print(f"{k=}, {loss_.shape=}")
            # print(f"{k=}, {grads_.shape=}")

        with torch.no_grad():
            losses = torch.cat(losses, dim=0).view(1,-1)
            grads_all = torch.cat(grads, dim=0)
            
            # class별로 변화량이 비슷하도록 normalize
            n_grads_all = F.normalize(grads_all, p=2, dim=1) # 4 * (weight&bias 차원수)
            n_r_new_grads = F.normalize(r_new_grads, p=2, dim=1) # (후보수) * (weight&bias 차원수)

        return losses, n_grads_all, n_r_new_grads



    def prepare_train_loader(self, task_id):
        """
        Compute gradient for memory replay
        Compute individual sample gradient (against buffer data) for all current data
            loader로 불러와서 모든 output과 embedding을 저장
            gradient 계산 (W, b)
        각 batch별 loss와 std를 가장 낮게 하는 (하나)의 sample만 취해서 학습에 사용
        Return train loader
        """
        num_workers = self.params.get('num_dataloader_workers', 0)
        if task_id == 1: # no memory
            return self.benchmark.load(task_id, self.params['batch_size_train'],
                                    num_workers=num_workers, pin_memory=True)[0]

        # self.non_select_indexes = list(range(12000))
        self.non_select_indexes = copy.deepcopy(self.benchmark.seq_indices_train[task_id])
        
        losses, n_grads_all, n_r_new_grads = self.get_loss_grad_all(task_id)
        loss_matrix = losses.repeat(len(n_r_new_grads), 1)
        forget_matrix = torch.matmul(n_r_new_grads, torch.transpose(n_grads_all, 0, 1))


        # current data selection
        accumulate_select_indexes = []
        accumulate_sum = []
        select_indexes = []
        
        # for debug
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=False,
                                   num_workers=num_workers, pin_memory=True)[0]
        targets = train_loader.dataset.targets \
            if hasattr(train_loader.dataset, "targets") else train_loader.dataset.dataset.targets
        num_dict = {x:0 for x in targets.unique().cpu().numpy()}
        num_dict_list = {x:list() for x in targets.unique().cpu().numpy()}

        # data_len = len(targets)
        data_len = len(loss_matrix)

        # for debugging
        classwise_loss = []
        inputs = train_loader.dataset.inputs \
            if hasattr(train_loader.dataset, "inputs") else train_loader.dataset.dataset.inputs
        optim = self.prepare_optimizer(task_id)
        crit = self.prepare_criterion(task_id)
        original_model = copy.deepcopy(self.backbone)

        for b in range(data_len-1):
            # inp, targ, t_id  = inp.to(device), targ.to(device), t_id.to(device)
            print(f"{loss_matrix.shape=}")
            loss_matrix = loss_matrix - self.params['alpha'] * forget_matrix
            loss_mean = torch.mean(loss_matrix, dim=1, keepdim=True)
            loss_std = torch.std(loss_matrix, dim=1, keepdim=True)

            select_ind = torch.argmin(loss_mean + loss_std, dim=0)
            accumulate_sum.append(copy.deepcopy(loss_mean[select_ind].item() + loss_std[select_ind].item()))
            classwise_loss.append(loss_matrix[select_ind].view(-1).detach().clone().cpu().numpy())
            print(f"{classwise_loss[-1]=}")

            target_idx = self.non_select_indexes[select_ind.item()]
            print(f"{targets[target_idx].item()=}")
            num_dict[targets[target_idx].item()] += 1
            for k in num_dict:
                num_dict_list[k].append(num_dict[k])

            select_indexes.append(self.non_select_indexes[select_ind.item()])
            accumulate_select_indexes.append(copy.deepcopy(select_indexes))
            
            # del self.benchmark.seq_indices_train[task_id][select_ind.item()]
            del self.non_select_indexes[select_ind.item()]


            # model update??
            inp = inputs[select_indexes]
            targ = targets[select_indexes]
            device = self.params['device']
            inp, targ  = inp.to(device), targ.to(device)
            self.training_step([task_id], inp, targ, optim, crit)
            loss_matrix, forget_matrix = self.get_loss_grad_all(task_id)

            # best_buffer_losses = loss_matrix[select_ind].view(1,-1)
            # loss_matrix = best_buffer_losses.repeat(len(n_new_grads)-1, 1)

            # n_new_grads = torch.cat((n_new_grads[:select_ind.item()], n_new_grads[select_ind.item()+1:]))
            # forget_matrix = torch.cat((forget_matrix[:select_ind.item()], forget_matrix[select_ind.item()+1:]))

        # for debugging
        save_path = f"./figs/alpha_{self.params['alpha']}_v2"
        os.makedirs(save_path, exist_ok=True)
        plt.plot(accumulate_sum)
        plt.savefig(f"{save_path}/tid_{task_id}_accumulate_loss.png")
        plt.clf()

        classwise_loss = np.array(classwise_loss).T
        for i, e in enumerate(classwise_loss):
            plt.plot(e, label=i)
        plt.legend(loc="best")
        plt.savefig(f"{save_path}/tid_{task_id}_classwise_loss.png")
        plt.clf()

        for i, e in num_dict_list.items():
            plt.plot(e, label=i)
        plt.legend(loc="best")
        plt.savefig(f"{save_path}/tid_{task_id}_class_num.png")
        plt.clf()

        self.backbone = copy.deepcopy(original_model)
        best_ind = np.argmin(np.array(accumulate_sum))
        select_curr_indexes = accumulate_select_indexes[best_ind]
        print(f"{len(select_curr_indexes)=}")

        select_curr_indexes = list(set(select_curr_indexes))
        self.benchmark.seq_indices_train[task_id] = select_curr_indexes
        
        num_workers = self.params.get('num_dataloader_workers', 0)
        return self.benchmark.load(task_id, self.params['batch_size_train'],
                                   num_workers=num_workers, pin_memory=True)[0]


    def training_step(self, task_ids, inp, targ, optimizer, criterion, sensitive=None):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_ids)
        loss = criterion(pred, targ)
        loss.backward()
        if task_ids[0] > 1:
            grad_batch = flatten_grads(self.backbone).detach().clone()
            optimizer.zero_grad()

            # get grad_ref
            inp_ref, targ_ref, task_ids_ref, sensitive_ref = self.sample_batch_from_memory()
            pred_ref = self.backbone(inp_ref, task_ids_ref)
            loss = criterion(pred_ref, targ_ref.reshape(len(targ_ref)))
            loss.backward()
            grad_ref = flatten_grads(self.backbone).detach().clone()
            grad_batch += self.params['tau']*grad_ref

            optimizer.zero_grad()
            self.backbone = assign_grads(self.backbone, grad_batch)
        optimizer.step()
