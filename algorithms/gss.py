# Copyright 2019-present, MILA, KU LEUVEN.
# All rights reserved.
# code imported and modified from https://github.com/RaptorMai/online-continual-learning/blob/main/utils/buffer/gss_greedy_update.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .baselines import BaseMemoryContinualAlgoritm

def get_grad_vector(pp, grad_dims, device):
    """
        gather the gradients in one vector
    """
    grads = torch.Tensor(sum(grad_dims)).to(device)
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads

def cosine_similarity(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    sim = torch.mm(x1, x2.t())/(w1 * w2.t()).clamp(min=eps)
    return sim

# python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve random \
# --update GSS --eps_mem_batch 10 --gss_mem_strength 20 --mem_size 5000

class GSSGreedy(BaseMemoryContinualAlgoritm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("GSSGreedy")
        # the number of gradient vectors to estimate new samples similarity, line 5 in alg.2
        self.params = args[2]
        self.mem_strength = self.params['batch_size_train']*2 # hyperparameter
        self.gss_batch_size = self.params['batch_size_train'] # Random sampling batch size to estimate score
        self.device = self.params['device']
        self.mem_size = self.params['per_task_memory_examples']
        self.buffer_score = torch.FloatTensor(self.mem_size).fill_(0).to(self.device)
        self._modify_benchmark()

    def _modify_benchmark(self):
        num_tasks = self.benchmark.num_tasks
        self.memory_current_index = dict()
        for task in range(1, num_tasks+1):
            self.benchmark.memory_indices_train[task] = list()
            self.memory_current_index[task] = 0

    def training_step(self, task_ids, inp, targ, indices, optimizer, criterion):
        super().training_step(task_ids, inp, targ, optimizer, criterion)
        self.update(inp, targ, task_ids[0].item(), indices)

    def update_memory_idx(self, task, insert_idx, remove_idx):
        """
        update self.benchmark.memory_indices_train[task]
        insert_idx: index to insert in self.benchmark.trains[task]
        remove_idx: target index to remove (index from memory)
        """
        for i, e in enumerate(remove_idx):
            # t = self.benchmark.memory_indices_train[task].index(e.item())
            self.benchmark.memory_indices_train[task][e] = insert_idx[i].item()

    def insert_memory_idx(self, task, insert_idx):
        """
        insert self.benchmark.memory_indices_train[task] (list)
        insert_idx: index to insert in self.benchmark.memory_indices_train[task]
        """
        self.benchmark.memory_indices_train[task].extend(insert_idx.cpu().numpy().tolist())

    def get_ith_memory(self, task, indices):
        # memory_dataset = Subset(self.benchmark.trains[task], self.benchmark.memory_indices_train[task])
        # partial_memory = Subset(memory_dataset, indices)
        partial_memory = Subset(self.benchmark.trains[task], indices)
        loader = DataLoader(partial_memory, len(indices), shuffle=False)
        memory_iter = iter(loader)
        batch = next(memory_iter)
        # inp, targ, task_id, idx, *_ = batch
        inp, targ, *_ = batch
        return inp.to(self.device), targ.to(self.device)

    def update(self, x, y, task_id, indices):
        self.backbone.eval()
        grad_dims = []
        for param in self.backbone.parameters():
            grad_dims.append(param.data.numel())

        place_left = self.mem_size - self.memory_current_index[task_id]
        if place_left <= 0:  # buffer is full
            batch_sim, mem_grads = self.get_batch_sim(grad_dims, x, y, task_id)
            if batch_sim < 0:
                buffer_score = self.buffer_score[:self.memory_current_index[task_id]]
                buffer_sim = (buffer_score - torch.min(buffer_score)) / \
                             ((torch.max(buffer_score) - torch.min(buffer_score)) + 0.01)
                # draw candidates for replacement from the buffer
                index = torch.multinomial(buffer_sim, x.size(0), replacement=False)
                # estimate the similarity of each sample in the recieved batch
                # to the randomly drawn samples from the buffer.
                batch_item_sim = self.get_each_batch_sample_sim(grad_dims, mem_grads, x, y)
                # normalize to [0,1]
                scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(1)
                buffer_repl_batch_sim = ((self.buffer_score[index] + 1) / 2).unsqueeze(1)
                # draw an event to decide on replacement decision
                outcome = torch.multinomial(torch.cat((scaled_batch_item_sim, buffer_repl_batch_sim), dim=1), 1,
                                            replacement=False)
                # replace samples with outcome =1
                added_indx = torch.arange(end=batch_item_sim.size(0)).to(self.device)
                sub_index = outcome.squeeze(1).bool()

                self.update_memory_idx(task_id, indices[added_indx[sub_index].cpu()], index[sub_index])
                self.buffer_score[index[sub_index]] = batch_item_sim[added_indx[sub_index]].clone()
        else:
            offset = min(place_left, x.size(0))
            x = x[:offset]
            y = y[:offset]
            indices = indices[:offset]
            # first buffer insertion
            if self.memory_current_index[task_id] == 0:
                batch_sample_memory_cos = torch.zeros(x.size(0)).to(self.device) + 0.1
            else:
                # draw random samples from buffer
                mem_grads = self.get_rand_mem_grads(grad_dims, task_id)
                # estimate a score for each added sample
                batch_sample_memory_cos = self.get_each_batch_sample_sim(grad_dims, mem_grads, x, y)
            self.insert_memory_idx(task_id, indices)
            self.buffer_score[self.memory_current_index[task_id]:self.memory_current_index[task_id] + offset] \
                .data.copy_(batch_sample_memory_cos)
            self.memory_current_index[task_id] += offset
        self.backbone.train()

    def get_batch_sim(self, grad_dims, batch_x, batch_y, task_id):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
            batch_x: batch images
            batch_y: batch labels
        Returns: score of current batch, gradient from memory subsets
        """
        mem_grads = self.get_rand_mem_grads(grad_dims, task_id)
        self.backbone.zero_grad()
        pred = self.backbone(batch_x)
        loss = F.cross_entropy(pred, batch_y)
        loss.backward()
        batch_grad = get_grad_vector(self.backbone.parameters, grad_dims, self.device).unsqueeze(0)
        batch_sim = max(cosine_similarity(mem_grads, batch_grad))
        return batch_sim, mem_grads

    def get_rand_mem_grads(self, grad_dims, task_id):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
        Returns: gradient from memory subsets
        """
        gss_batch_size = min(self.gss_batch_size, self.memory_current_index[task_id])
        num_mem_subs = min(self.mem_strength, self.memory_current_index[task_id] // gss_batch_size)
        mem_grads = torch.zeros(num_mem_subs, sum(grad_dims), dtype=torch.float32).to(self.device)
        shuffeled_inds = torch.randperm(self.memory_current_index[task_id]).to(self.device)
        for i in range(num_mem_subs):
            random_batch_inds = shuffeled_inds[
                                i * gss_batch_size:i * gss_batch_size + gss_batch_size]
            batch_x, batch_y = self.get_ith_memory(task_id, random_batch_inds)
            self.backbone.zero_grad()
            loss = F.cross_entropy(self.backbone.forward(batch_x), batch_y)
            loss.backward()
            mem_grads[i].data.copy_(get_grad_vector(self.backbone.parameters, grad_dims, self.device))
        return mem_grads

    def get_each_batch_sample_sim(self, grad_dims, mem_grads, batch_x, batch_y):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
            mem_grads: gradient from memory subsets
            batch_x: batch images
            batch_y: batch labels
        Returns: score of each sample from current batch
        """
        cosine_sim = torch.zeros(batch_x.size(0)).to(self.device)
        for i, (x, y) in enumerate(zip(batch_x, batch_y)):
            self.backbone.zero_grad()
            ptloss = F.cross_entropy(self.backbone.forward(x.unsqueeze(0)), y.unsqueeze(0))
            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine similarity
            this_grad = get_grad_vector(self.backbone.parameters, grad_dims, self.device).unsqueeze(0)
            cosine_sim[i] = max(cosine_similarity(mem_grads, this_grad))
        return cosine_sim