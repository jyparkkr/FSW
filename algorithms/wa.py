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
        self.T = 2 # temperature
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
            elif "Bert" in self.backbone.__class__.__name__:
                if hasattr(self.backbone, "classifier"):
                    weights = self.backbone.classifier.weight.data
                else:
                    raise NotImplementedError
            else:
                print(f"{self.backbone.__class__.__name__=}")
                raise NotImplementedError
            
            norm_li = []
            for w in weights:
                norm_li.append(torch.norm(w).item())
            norm_li = np.array(norm_li)

            buf_target_classes = self.benchmark.class_idx[:(self.current_task-1)*self.benchmark.num_classes_per_split]
            cur_target_classes = self.benchmark.class_idx[
                (self.current_task-1)*self.benchmark.num_classes_per_split:self.current_task*self.benchmark.num_classes_per_split]

            mean_old = np.mean(norm_li[buf_target_classes])
            mean_new = np.mean(norm_li[cur_target_classes])
            
            ratio = mean_old/mean_new
            
            weights[cur_target_classes] = torch.nn.Parameter(ratio * weights[cur_target_classes])
            
            print(f"{mean_old/mean_new=}")
        super().training_task_end()

    def training_step(self, task_ids, inp, targ, optimizer, criterion, sample_weight=None, sensitive_label=None):
        optimizer.zero_grad()
        if (task_ids[0] > 1) and self.params['tau']: # tau is 1 for WA
            inp_ref, targ_ref, task_ids_ref = self.sample_batch_from_memory()
            if isinstance(inp, list):
                inp = [torch.cat((inp[i], e), dim=0) for i, e in enumerate(inp_ref)]
                # task_ids = [torch.cat((task_ids[i], e), dim=0) for i, e in enumerate(task_ids_ref)]
            else:
                inp = torch.cat((inp, inp_ref), dim=0)
                # task_ids = torch.cat((task_ids, task_ids_ref), dim=0)
                
        # prob = self.backbone(inp, task_ids)
        prob = self.backbone(inp)
        if (task_ids[0] > 1) and self.params['tau']: # tau is 1 for WA
            if isinstance(targ, np.ndarray):
                targ = np.concatenate((targ, targ_ref), axis=0)
            elif isinstance(targ, torch.Tensor):
                targ = torch.cat((targ, targ_ref), dim=0)
            elif isinstance(targ, list):
                targ.extend(targ_ref)
            else:
                raise NotImplementedError
            
        loss = (1 - self.lamb) * criterion(prob, targ)

        if (task_ids[0] > 1) and self.params['tau']: # tau is 1 for WA
            buf_target_classes = self.benchmark.class_idx[:(self.current_task-1)*self.benchmark.num_classes_per_split]
            prob_norm = torch.log_softmax(prob[:,buf_target_classes]/self.T, dim=1)
            og_prob = self.og_backbone(inp)
            og_prob_norm = torch.softmax(og_prob[:,buf_target_classes]/self.T, dim=1)
            kd_loss = -1 * torch.mul(og_prob_norm, prob_norm).sum() / prob_norm.shape[0]
            loss += self.lamb * kd_loss
        loss.backward()
        optimizer.step()



