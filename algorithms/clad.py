import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

import copy
from cl_gym.algorithms.utils import flatten_grads, assign_grads
from .baselines import BaseContinualAlgoritm

class FastDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class CLAD(BaseContinualAlgoritm):
    def __init__(self, backbone, benchmark, params, requires_memory=True):
        super().__init__(backbone, benchmark, params, requires_memory=requires_memory)
        self.p = 0.5 # proportion of conflict classes
        self.eta = params.get("eta", 0.01) # hyperparameter 0.01, 0.1, 1
        if not hasattr(self, "embedding_size"):
            if self.params['model'] == "MLP":
                self.embedding_size = 256
            elif self.params['model'] == "resnet18small":
                self.embedding_size = 160
            elif self.params['model'] == "resnet18":
                self.embedding_size = 1024
            elif self.params['model'] == "bert":
                self.embedding_size = 768
            else:
                raise NotImplementedError
        if "Bert" in self.backbone.__class__.__name__:
            self.bert = True
        else:
            self.bert = False

        print(f"CLAD")

    def before_training_task(self):
        # called before loader, optimizer, criterion initialized
        super().before_training_task()
        if self.current_task > 1:
            device = self.params['device']
            self.og_backbone = copy.deepcopy(self.backbone).to(device)
            self.og_backbone.eval()

            n_class = len(self.benchmark.class_idx)
            
            print(f"{device=}")

            with torch.no_grad():
                all_logit = torch.zeros(n_class)
                self.backbone.eval()
                print(f"{self.current_task=}")
                train_loader = self.prepare_train_loader(self.current_task)
                for items in train_loader:
                    item_to_devices = [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]
                    inp, targ, task_ids, *_ = item_to_devices
                    if isinstance(inp, list):
                        inp = [x.to(device) for x in inp]
                    output = self.og_backbone(inp)
                    logit = torch.mean(output, dim=0)
                    all_logit += logit.detach().cpu()
                        
            # k = number of old classes * proportion of conflict classes
            k = int((int(n_class/self.params['num_tasks']) * (self.current_task-1)) * self.p)
            
            # get list of conflict classes
            _, c = torch.topk(all_logit, k)
            c = c.to(device, dtype=torch.int64)
            print("c:", c)

            conflict_x_buffer = list()
            conflict_y_buffer = list()
            for t in range(1, self.current_task):
                conflict_inds = list()
                task_dataset = self.benchmark.trains[t]
                memory_indices = self.benchmark.memory_indices_train[t]
                for ind in memory_indices:
                    if task_dataset.targets[ind] in c:
                        conflict_inds.append(ind)
                data, y = task_dataset.getitem_test_transform_list(conflict_inds)
                if len(data) == 0:
                    continue
                data = self.data_to_tensor(data)
                print(f"{data.shape=}")
                y = torch.tensor(y)
                conflict_x_buffer.append(data)
                conflict_y_buffer.append(y)

            conflict_x_buffer = torch.cat(conflict_x_buffer, 0)
            conflict_y_buffer = torch.cat(conflict_y_buffer, 0)


            conflict_buffer_ds = FastDataset(conflict_x_buffer, conflict_y_buffer)
            conflict_buffer_loader = DataLoader(conflict_buffer_ds, batch_size=self.params['batch_size_memory'], shuffle=False)

            with torch.no_grad():
                conflict_old_embs = torch.zeros(self.embedding_size)
                num = 0
                for x_buff, y_buff in conflict_buffer_loader:
                    if isinstance(x_buff, list):
                        x_buff = [x.to(device) for x in x_buff]
                    else:
                        x_buff = x_buff.to(device)

                    output, emb = self.og_backbone.forward_embeds(x_buff)
                    avg_emb = torch.mean(emb, dim=0)
                    conflict_old_embs += avg_emb.detach().cpu()
                    num += 1
                    
            conflict_old_embs = conflict_old_embs / num
            self.conflict_old_embs = conflict_old_embs.to(device, dtype=torch.float32)
            self.c = c

    def training_task_end(self):
        print("training_task_end")
        super().training_task_end()

    def data_to_tensor(self, data):
        if not self.bert:
            if isinstance(data[0], np.ndarray):
                data = [torch.from_numpy(cand) for cand in data]
            data = torch.stack(data, 0)
        else:
            if isinstance(data[0][0], np.ndarray):
                data = [[torch.from_numpy(cand) for cand in x] for x in data]
            data = [torch.stack(x, 0) for x in data]
            data = torch.stack(data, 0)
        return data

    def training_step(self, task_ids, inp, targ, optimizer, criterion, sample_weight=None, sensitive_label=None):
        optimizer.zero_grad()
        pred, embs1 = self.backbone.forward_embeds(inp, task_ids)
        criterion.reduction = "none"
        loss = criterion(pred, targ)
        criterion.reduction = "mean"
        if sample_weight is not None:
            loss = loss*sample_weight
            # print(f"{loss=}")
            # print(f"{loss.shape=}")
            # print(f"{sample_weight.shape=}")
        loss1 = loss.mean()
        
        if (task_ids[0] > 1) and self.params['tau']:
            inp_ref, targ_ref, task_ids_ref = self.sample_batch_from_memory()
            pred_ref, embs2 = self.backbone.forward_embeds(inp_ref, task_ids_ref)
            loss2 = criterion(pred_ref, targ_ref.reshape(len(targ_ref)))

            #added
            conflict_inds = []
            for ind in range(len(targ_ref)):
                if targ_ref[ind] in self.c:
                    conflict_inds.append(ind)
                    
            conflict_embs2 = embs2[conflict_inds]

            # online loss
            loss_on = 1 + nn.functional.cosine_similarity(torch.mean(embs1, dim=0), torch.mean(conflict_embs2, dim=0), dim=0)
            # offline loss
            loss_off = 1 + nn.functional.cosine_similarity(torch.mean(embs1, dim=0), torch.mean(self.conflict_old_embs, dim=0), dim=0)
            # clad loss = online loss + offline loss
            loss_clad = loss_on + loss_off
        
            loss = (loss1 + self.params['tau'] * loss2) + self.eta * loss_clad
        else:
            loss = loss1

        loss.backward()
        optimizer.step()
