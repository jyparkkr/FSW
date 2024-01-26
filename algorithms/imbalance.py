import torch
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.agem import AGEM
from cl_gym.algorithms.utils import flatten_grads, assign_grads
from torch.nn.functional import relu, avg_pool2d
import matplotlib.pyplot as plt

import copy
import os

def bool2idx(arr):
    idx = list()
    for i, e in enumerate(arr):
        if e == 1:
            idx.append(i)
    return np.array(idx)


class Heuristic2(ContinualAlgorithm):
    # Implementation is partially based on: https://github.com/MehdiAbbanaBennani/continual-learning-ogdplus
    def __init__(self, backbone, benchmark, params, **kwargs):
        self.backbone = backbone
        self.benchmark = benchmark
        self.params = params
        
        super(Heuristic2, self).__init__(backbone, benchmark, params, **kwargs)

    def memory_indices_selection(self, task):
        ## update self.benchmark.memory_indices_train[task] with len self.benchmark.per_task_memory_examples
        
        indices_train = np.arange(self.per_task_memory_examples)
        # num_examples = self.benchmark.per_task_memory_examples
        # indices_train = self.sample_uniform_class_indices(self.trains[task], start_cls, end_cls, num_examples)
        # # indices_test = self.sample_uniform_class_indices(self.tests[task], start_cls, end_cls, num_examples)
        assert len(indices_train) == self.per_task_memory_examples
        self.benchmark.memory_indices_train[task] = indices_train[:]

    def update_episodic_memory(self):
        # self.memory_indices_selection(self.current_task)
        self.episodic_memory_loader, _ = self.benchmark.load_memory_joint(self.current_task,
                                                                          batch_size=self.params['batch_size_memory'],
                                                                          shuffle=True,
                                                                          pin_memory=True)
        self.episodic_memory_iter = iter(self.episodic_memory_loader)


    def sample_batch_from_memory(self):
        try:
            batch = next(self.episodic_memory_iter)
        except StopIteration:
            self.episodic_memory_iter = iter(self.episodic_memory_loader)
            batch = next(self.episodic_memory_iter)
        
        device = self.params['device']
        inp, targ, task_id, *_ = batch
        return inp.to(device), targ.to(device), task_id.to(device), _

    def training_task_end(self):
        """
        Select what to store in the memory in this step
        """
        print("training_task_end")
        if self.requires_memory:
            self.update_episodic_memory()
        self.current_task += 1

    def forward_embeds(self, inp):
        if self.params['dataset'] in ['MNIST', "FMNIST"]: #MLP
            inp = inp.view(inp.shape[0], -1)
            out = inp
            for block in self.backbone.blocks:
                embeds = out
                out = block(out)
            return out, embeds
        else: #ResNet
            bsz = inp.size(0)
            shape = (bsz, self.backbone.dim, \
                        self.backbone.input_shape[-2], self.backbone.input_shape[-1])
            out = relu(self.backbone.bn1(self.backbone.conv1(inp.view(shape))))
            out = self.backbone.layer1(out)
            out = self.backbone.layer2(out)
            out = self.backbone.layer3(out)
            out = self.backbone.layer4(out)
            out = avg_pool2d(out, 4)
            embeds = out.view(out.size(0), -1)
            out = self.backbone.linear(embeds)
            return out, embeds

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
        # if current_set:
        #     for k in classwise_loss:
        #         if len(classwise_loss[k]) == 0:
        #             print(f"{k} is missing")
        #             for kk in classwise_loss:
        #                 print(f"{len(classwise_loss[kk])=}")
        #             # classwise_loss[k].append(torch.empty((0, self.pred_shape*(self.embeds_shape+1)), device=device, dtype=torch.float32))
        #             # classwise_loss[k].append(torch.empty((0, self.pred_shape*(self.embeds_shape+1)), device=device, dtype=torch.float32))
        #             classwise_loss[k].append(torch.zeros((0, self.pred_shape*(self.embeds_shape+1)), device=device, dtype=torch.float32))
        #             classwise_grad_dict[k].append(torch.zeros((0, self.pred_shape*(self.embeds_shape+1)), device=device, dtype=torch.float32))

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
        # r_new_grads = new_grads
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


    def converter(self, losses, alpha, grads_all, new_grads):
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
        return dg, c



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
        self.non_select_indexes = list(range(len(self.benchmark.seq_indices_train[task_id])))
        
        losses, n_grads_all, n_r_new_grads = self.get_loss_grad_all(task_id)
        A, b = self.converter(losses, self.params['alpha'], n_grads_all, n_r_new_grads)

        from algorithms.optimization.cplex_solver import LS_solver
        np_weight = torch.ones(A.shape[1])*0.9
        weight = LS_solver(A.cpu().detach().numpy(), b.view(-1).cpu().detach().numpy())
        np_weight = torch.tensor(np.array(weight))
        # np_weight = torch.ones(A.shape[1])
        # np_weight = torch.ones(A.shape[1])*0.5
        print(f"{np_weight=}")
        print(f"{np_weight.shape=}")

        self.benchmark.update_sample_weight(task_id, np_weight)

        print(f"{np_weight.sum()=}")
        print(f"{np_weight.mean()=}")
        print(f"{np_weight.max()=}")
        print(f"{np_weight.min()=}")
        print(f"{np_weight.min()=}")

        # # for debugging
        # save_path = f"./figs/alpha_{self.params['alpha']}_v2"
        # os.makedirs(save_path, exist_ok=True)
        # plt.hist(weight, color = "green", alpha = 0.4, bins = 100, edgecolor="black")
        # plt.savefig(f"{save_path}/tid_{task_id}_weight_distribution.png")
        # plt.clf()
        
        num_workers = self.params.get('num_dataloader_workers', 0)
        return self.benchmark.load(task_id, self.params['batch_size_train'],
                                   num_workers=num_workers, pin_memory=True)[0]


    def training_step(self, task_ids, inp, targ, optimizer, criterion, sample_weight=None):
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
        if task_ids[0] > 1:
            grad_batch = flatten_grads(self.backbone).detach().clone()
            optimizer.zero_grad()

            # get grad_ref
            inp_ref, targ_ref, task_ids_ref, sample_weight_ref = self.sample_batch_from_memory()
            pred_ref = self.backbone(inp_ref, task_ids_ref)
            loss = criterion(pred_ref, targ_ref.reshape(len(targ_ref)))
            loss.backward()
            grad_ref = flatten_grads(self.backbone).detach().clone()
            grad_batch += self.params['lambda']*grad_ref

            optimizer.zero_grad()
            self.backbone = assign_grads(self.backbone, grad_batch)
        optimizer.step()
