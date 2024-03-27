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
import time

import copy
import os

def bool2idx(arr):
    idx = list()
    for i, e in enumerate(arr):
        if e == 1:
            idx.append(i)
    return np.array(idx)

class Heuristic(ContinualAlgorithm):
    # Implementation is partially based on: https://github.com/MehdiAbbanaBennani/continual-learning-ogdplus
    def __init__(self, backbone, benchmark, params, **kwargs):
        self.backbone = backbone
        self.benchmark = benchmark
        self.params = params
        super(Heuristic, self).__init__(backbone, benchmark, params, **kwargs)

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
        if "MNIST" in self.params['dataset']: #MLP
            inp = inp.view(inp.shape[0], -1)
            out = inp
            for block in self.backbone.blocks:
                embeds = out
                out = block(out)
            return out, embeds
        elif "ResNet18Small" in str(self.backbone.__class__):
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
        else:
            if hasattr(self.backbone, "forward_embeds"):
                return self.backbone.forward_embeds(inp)
            raise NotImplementedError

    def get_loss_grad(self):
        raise NotImplementedError
    
    def get_loss_grad_all(self):
        raise NotImplementedError

    def converter(self):
        raise NotImplementedError

    def training_step(self):
        raise NotImplementedError

    def prepare_train_loader(self, task_id, solver=None, epoch=0):
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
        
        if self.params['alpha'] == 0:
            return self.benchmark.load(task_id, self.params['batch_size_train'],
                                    num_workers=num_workers, pin_memory=True)[0]
        
        if epoch <= 1:
            self.original_seq_indices_train = self.benchmark.seq_indices_train[task_id]
            if hasattr(self.benchmark.trains[task_id], "sensitive"):
                print(f"Num. of sensitives: {(self.benchmark.trains[task_id].sensitive[self.original_seq_indices_train] != self.benchmark.trains[task_id].targets[self.original_seq_indices_train]).sum().item()}")
        else:
            self.benchmark.seq_indices_train[task_id] = copy.deepcopy(self.original_seq_indices_train)
        self.non_select_indexes = list(range(len(self.benchmark.seq_indices_train[task_id])))

        losses, n_grads_all, n_r_new_grads = self.get_loss_grad_all(task_id) 
        # n_grads_all: (class_num) * (weight&bias 차원수)
        # n_r_new_grads: (current step data 후보수) * (weight&bias 차원수)

        print(f"{losses=}")
        # print(f"{n_grads_all.mean(dim=1)=}")

        A, b = self.converter(losses, self.params['alpha'], n_grads_all, n_r_new_grads)
        A_np = A.cpu().detach().numpy().astype('float64')
        b_np = b.view(-1).cpu().detach().numpy().astype('float64')

        print(f"{A_np.shape=}")
        print(f"{b_np.shape=}")

        i = time.time()
        weight = solver(A_np, b_np)
        
        print(f"Elapsed time:{np.round(time.time()-i, 3)}")
        print(f"Loss difference:{np.matmul(A_np, weight)-b_np}")

        tensor_weight = torch.tensor(np.array(weight), dtype=torch.float32)
        self.benchmark.update_sample_weight(task_id, tensor_weight)

        # Need to update self.benchmark.seq_indices_train[task] - to ignore weight = 0
        drop_threshold = 0.05
        updated_seq_indices = np.array(self.benchmark.seq_indices_train[task_id])[np.array(weight)>drop_threshold]
        self.benchmark.seq_indices_train[task_id] = updated_seq_indices.tolist()
        print(f"{len(updated_seq_indices)=}")
        if hasattr(self.benchmark.trains[task_id], "sensitive"):
            print(f"sensitive samples / selected samples = {(self.benchmark.trains[task_id].sensitive[updated_seq_indices] != self.benchmark.trains[task_id].targets[updated_seq_indices]).sum().item()} / {len(updated_seq_indices)}")


        # for debugging
        os.makedirs(f"{self.params['output_dir']}/figs", exist_ok=True)
        plt.hist(weight, color = "green", alpha = 0.4, bins = 100, edgecolor="black")
        plt.axvline(drop_threshold, color='black', linestyle='dashed')
        plt.savefig(f"{self.params['output_dir']}/figs/tid_{task_id}_epoch_{epoch}_weight_distribution.png")
        plt.clf()
        
        return self.benchmark.load(task_id, self.params['batch_size_train'],
                                   num_workers=num_workers, pin_memory=True)[0]
