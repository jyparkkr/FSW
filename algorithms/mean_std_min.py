import torch
import numpy as np
import copy

from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.agem import AGEM
from cl_gym.algorithms.utils import flatten_grads, assign_grads
from torch.nn.functional import relu, avg_pool2d

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

    def prepare_train_loader(self, task_id):
        """
        Compute gradient for memory replay
        Compute individual sample gradient (against buffer data) for all current data
            loader로 불러와서 모든 output과 embedding을 저장
            gradient 계산 (W, b)
        각 batch별 loss와 std를 가장 낮게 하는 (하나)의 sample만 취해서 학습에 사용
        Return train loader
        """
        if task_id == 1: # no memory
            num_workers = self.params.get('num_dataloader_workers', 0)
            return self.benchmark.load(task_id, self.params['batch_size_train'],
                                    num_workers=num_workers, pin_memory=True)[0]

        device = self.params['device']
        criterion = self.prepare_criterion(task_id)

        # 10 대신 받아오는걸로 바꿔야함
        classwise_loss = {x:list() for x in range(10)}
        classwise_bias_grad_dict = {x:list() for x in range(10)}
        classwise_weight_grad_dict = {x:list() for x in range(10)}

        # Compute gradient for memory replay
        for batch_idx, (inp, targ, t_id, *_) in enumerate(self.episodic_memory_loader):
            # self.backbone.forward
            inp, targ, t_id  = inp.to(device), targ.to(device), t_id.to(device)
            # print(f"{inp.shape}")
            # print(f"{targ=}")
            # print(f"{targ[0].cpu().item()=}")

            pred, embeds = self.forward_embeds(inp)
            criterion.reduction = "none"
            loss = criterion(pred, targ.reshape(-1))
            criterion.reduction = "mean"
            
            buffer_bias_grads = torch.autograd.grad(loss.sum(), pred)[0]
            buffer_bias_expand = torch.repeat_interleave(buffer_bias_grads, embeds.shape[1], dim=1)
            buffer_weight_grads = buffer_bias_expand * embeds.repeat(1, pred.shape[1])

            for i, e in enumerate(targ):
                classwise_loss[e.cpu().item()].append(loss[i])
                classwise_bias_grad_dict[e.cpu().item()].append(buffer_bias_grads[i])
                classwise_weight_grad_dict[e.cpu().item()].append(buffer_weight_grads[i])
            self.backbone.zero_grad()

        # Compute gradient for current incremental step
        num_workers = self.params.get('num_dataloader_workers', 0)
        train_loader = self.benchmark.load(task_id, self.params['batch_size_train'], shuffle=False,
                                   num_workers=num_workers, pin_memory=True)[0]
        
        # parameter 받아오는걸로 바꾸어야함
        new_bias_grads = torch.empty((0, pred.shape[1]), device=device, dtype=torch.float32)
        new_weight_grads = torch.empty((0, pred.shape[1]*embeds.shape[1]), device=device, dtype=torch.float32)
        for batch_idx, (inp, targ, t_id, *_) in enumerate(train_loader):
            inp, targ, t_id  = inp.to(device), targ.to(device), t_id.to(device)

            pred, embeds = self.forward_embeds(inp)
            criterion.reduction = "none"
            loss = criterion(pred, targ.reshape(-1))
            criterion.reduction = "mean"
            current_bias_grads = torch.autograd.grad(loss.sum(), pred)[0]
            current_bias_expand = torch.repeat_interleave(current_bias_grads, embeds.shape[1], dim=1)
            current_weight_grads = current_bias_expand * embeds.repeat(1, pred.shape[1])

            for i, e in enumerate(targ):
                classwise_loss[e.cpu().item()].append(loss[i])
                classwise_bias_grad_dict[e.cpu().item()].append(current_bias_grads[i])
                classwise_weight_grad_dict[e.cpu().item()].append(current_weight_grads[i])
            self.backbone.zero_grad()

            new_bias_grads = torch.cat([new_bias_grads, current_bias_grads])
            new_weight_grads = torch.cat([new_weight_grads, current_weight_grads])
            

        losses, bias_grads, weight_grads = [], [], []
        for k, v in classwise_loss.items():
            if len(v):
                v1 = classwise_bias_grad_dict[k]
                v2 = classwise_weight_grad_dict[k]
                losses.append(torch.stack(v).mean(dim=0).view(1, -1))
                bias_grads.append(torch.stack(v1).mean(dim=0).view(1, -1))
                weight_grads.append(torch.stack(v2).mean(dim=0).view(1, -1))

        losses = torch.cat(losses, dim=0).view(1,-1)
        with torch.no_grad():
            bias_grads = torch.cat(bias_grads, dim=0)
            weight_grads = torch.cat(weight_grads, dim=0)

            buffer_grads = torch.cat((bias_grads, weight_grads), dim=1)
            n_buffer_grads = F.normalize(buffer_grads, p=2, dim=1)

            new_grads = torch.cat((new_bias_grads, new_weight_grads), dim=1)
            new_grads_origin = new_grads.clone()
            n_new_grads = F.normalize(new_grads, p=2, dim=1)

            loss_matrix = losses.repeat(len(new_grads), 1).to(device)
            loss_matrix_origin = loss_matrix.clone()
            forget_matrix = torch.matmul(n_new_grads, torch.transpose(n_buffer_grads, 0, 1)).to(device)


        # n_bias_grad = F.normalize(bias_grads, p=2, dim=1)
        # n_weight_grads = F.normalize(weight_grads, p=2, dim=1)
        print(f"{new_bias_grads.shape=}")
        print(f"{new_weight_grads.shape=}")
        print(f"{forget_matrix.shape=}")

        # print(f"{bias_grads.shape}")
        # print(f"{weight_grads.shape}")
        # print(f"{buffer_grads.shape}")
        # print(f"{torch.cat((n_bias_grad, n_weight_grads), dim=1)=}")
        # print(f"{n_buffer_grads=}")

        # current data selection
        accumulate_select_indexes = []
        accumulate_sum = []
        select_indexes = []

        targets = train_loader.dataset.targets \
            if hasattr(train_loader.dataset, "targets") else train_loader.dataset.dataset.targets
        print(f"{targets.shape=}")
        data_len = len(targets)
        non_select_indexes = list(range(data_len))
        num_dict = {x:0 for x in targets.unique().cpu().numpy()}
        for b in range(data_len):
            # inp, targ, t_id  = inp.to(device), targ.to(device), t_id.to(device)
            loss_matrix = loss_matrix - self.params['alpha'] * forget_matrix
            loss_mean = torch.mean(loss_matrix, dim=1, keepdim=True)
            loss_std = torch.std(loss_matrix, dim=1, keepdim=True)

            select_ind = torch.argmin(loss_mean + loss_std, dim=0)
            accumulate_sum.append(copy.deepcopy(loss_mean[select_ind].item() + loss_std[select_ind].item()))
            # print(f"{select_ind.item()=}")
            # print(f"{targets[select_ind.item()]=}")
            num_dict[targets[select_ind.item()].item()] += 1

            select_indexes.append(non_select_indexes[select_ind.item()])
            accumulate_select_indexes.append(copy.deepcopy(select_indexes))
            del non_select_indexes[select_ind.item()]

            best_buffer_losses = loss_matrix[select_ind].view(1,-1)
            loss_matrix = best_buffer_losses.repeat(len(n_new_grads)-1, 1).to(device)

            n_new_grads = torch.cat((n_new_grads[:select_ind.item()], n_new_grads[select_ind.item()+1:]))
            forget_matrix = torch.cat((forget_matrix[:select_ind.item()], forget_matrix[select_ind.item()+1:]))

        best_ind = np.argmin(np.array(accumulate_sum))
        print(f"{best_ind=}")
        select_curr_indexes = accumulate_select_indexes[best_ind]
        print(f"{len(select_curr_indexes)=}")
        print(f"{len(accumulate_select_indexes)=}")

        select_curr_indexes = list(set(select_curr_indexes))
        self.benchmark.seq_indices_train[task_id] = select_curr_indexes
        self.benchmark.seq_indices_test[task_id] = range(len(self.benchmark.tests[task_id]))
        
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
            grad_batch += self.params['lambda']*grad_ref

            optimizer.zero_grad()
            self.backbone = assign_grads(self.backbone, grad_batch)
        optimizer.step()
