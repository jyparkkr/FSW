import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np

import copy
from cl_gym.algorithms.utils import flatten_grads, assign_grads
from .baselines import BaseContinualAlgoritm

class WA(BaseContinualAlgoritm):
    def __init__(self, backbone, benchmark, params, requires_memory=True):
        super().__init__(backbone, benchmark, params, requires_memory=requires_memory)
        self.T = 2
        print(f"Weight Aligning")

    def before_training_task(self):
        # called before loader, optimizer, criterion initialized
        self.og_backbone = copy.deepcopy(self.backbone)
        self.og_backbone.eval()
        self.lamb = (self.current_task-1)/(self.current_task)

    def training_task_end(self):
        print("training_task_end")
        if self.current_task > 1:
            params = list()
            if "MLP" in self.backbone.__class__.__name__:
                weights =  self.backbone.blocks[-1].layers[0].weight.data
            elif "ResNet" in self.backbone.__class__.__name__:
                if hasattr(self.backbone, "fc"):
                    weights = self.backbone.fc.weight.data
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            
            norm_li = []
            for w in weights:
                norm_li.append(torch.norm(w).item())
            norm_li = np.array(norm_li)
            # task당 들어있는 class의 개수가 2개씩인 경우
            # CIFAR-100 같은 경우는 2를 10으로 바꾸면 될듯
            buf_target_classes = self.benchmark.class_idx[:(self.current_task-1)*self.benchmark.num_classes_per_split]
            cur_target_classes = self.benchmark.class_idx[
                (self.current_task-1)*self.benchmark.num_classes_per_split:self.current_task*self.benchmark.num_classes_per_split]

            mean_old = np.mean(norm_li[buf_target_classes])
            mean_new = np.mean(norm_li[cur_target_classes])
            
            # 논문에서 주장하기로는 이 ratio 값이 1보다 작게 나와서 current에 해당하는 weight를 줄여야 되는데
            # 실험해보면 1보다 작게 나오는 경우도 있지만 1 근처로 나오는 경우도 있음
            ratio = mean_old/mean_new
            
            # task당 들어있는 class의 개수가 2개씩인 경우
            # CIFAR-100 같은 경우는 2를 10으로 바꾸면 될듯
    #         if model is neural network
            weights[cur_target_classes] = torch.nn.Parameter(ratio * weights[cur_target_classes])
            # if model is resnet
    #         self.model.fc.weight.data[task*2:task*2+2] = torch.nn.Parameter(ratio * self.model.fc.weight.data[task*2:task*2+2])
            
            print(f"{mean_old/mean_new=}")
        super().training_task_end()
        # if self.requires_memory:
        #     self.update_episodic_memory()
        # self.current_task += 1

    def training_step(self, task_ids, inp, targ, optimizer, criterion, sample_weight=None, sensitive_label=None):
        optimizer.zero_grad()
        if (task_ids[0] > 1) and self.params['tau']: # tau is 1 for WA
            inp_ref, targ_ref, task_ids_ref = self.sample_batch_from_memory()
            inp = torch.cat((inp, inp_ref), dim=0)
        pred = self.backbone(inp)
        if (task_ids[0] > 1) and self.params['tau']: # tau is 1 for WA
            if isinstance(targ, np.ndarray):
                targ = np.concatenate((targ, targ_ref), axis=0)
            elif isinstance(targ, torch.Tensor):
                targ = torch.cat((targ, targ_ref), dim=0)
            elif isinstance(targ, list):
                targ.extend(targ_ref)
            else:
                raise NotImplementedError
            
        loss = (1 - self.lamb) * criterion(pred, targ)

        if (task_ids[0] > 1) and self.params['tau']: # tau is 1 for WA
            buf_target_classes = self.benchmark.class_idx[:(self.current_task-1)*self.benchmark.num_classes_per_split]
            pred_norm = torch.log_softmax(pred[:,buf_target_classes]/self.T, dim=1)
            og_pred = self.og_backbone(inp)
            og_pred_norm = torch.softmax(og_pred[:,buf_target_classes]/self.T, dim=1)
            kd_loss = -1 * torch.mul(og_pred_norm, pred_norm).sum() / pred_norm.shape[0]
            loss += self.lamb * kd_loss
        loss.backward()
        optimizer.step()



