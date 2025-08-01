# code imported and modified from https://github.com/jaehong31/OCS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import copy
from .baselines import BaseMemoryContinualAlgoritm


coreset_methods = ['uniform', 'coreset',
           'kmeans_features', 'kcenter_features', 'kmeans_grads',
           'kmeans_embedding', 'kcenter_embedding', 'kcenter_grads',
           'entropy', 'hardest', 'frcl', 'icarl', 'grad_matching']

class Coreset(torch.utils.data.Dataset):
    def __init__(self, set_size, input_shape=[784]):
        data_shape = [set_size]+input_shape

        self.data = torch.zeros(data_shape)
        self.targets = torch.ones((set_size))*-1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y

class FastDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.islist = True if isinstance(self.x, list) else False
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if self.islist:
            return [self.x[0][idx], self.x[1][idx], self.x[2][idx]], self.y[idx]
        return self.x[idx], self.y[idx]
    
def fast_cifar_loader(loaders, task_id, eval=True, device='cpu', bert=False):
    trains, evals = [], []
    if eval:
        train_loader, eval_loader = loaders
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            trains.append([data, target, task_id])

        for data, target in eval_loader:
            data = data.to(device)
            target = target.to(device)
            evals.append([data, target, task_id])
        return trains, evals
    else:
        for data, target in loaders:
            data = data.to(device)
            target = target.to(device)
            trains.append([data, target, task_id])
        return trains

class OCS(BaseMemoryContinualAlgoritm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the number of gradient vectors to estimate new samples similarity, line 5 in alg.2
        self.params = args[2]
        self.select_type = "ocs_select"
        self.ocspick = True

        # self.params['batch_size_train'] is actually stream size
        self.stream_size = self.params['batch_size_train']
        self.batch_size = self.stream_size//2 # based on github code
        self.ref_hyp = self.params['tau']
        self.r2c_iter = 100
        self.is_r2c = True
        self.tau = 1000.0
        
        self.memory_size = self.params['per_task_memory_examples'] * self.params['num_tasks']
        self.device = self.params['device']

        if "Bert" in self.backbone.__class__.__name__:
            self.bert = True
        else:
            self.bert = False

        self._modify_benchmark()

    def _modify_benchmark(self):
        num_tasks = self.benchmark.num_tasks
        self.memory_current_index = dict()
        for task in range(1, num_tasks+1):
            self.benchmark.memory_indices_train[task] = list()
            self.memory_current_index[task] = 0

    # this is not called in OCS
    def training_step(self, task_ids, inp, targ, indices, optimizer, criterion):
        super().training_step(task_ids, inp, targ, optimizer, criterion)

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

    def modify_memory_idx(self, task, modify_idx):
        """
        modify self.benchmark.memory_indices_train[task] (list)
        modify_idx: index to insert in self.benchmark.memory_indices_train[task]
        """
        if isinstance(modify_idx, torch.Tensor):
            modify_idx = modify_idx.cpu().numpy().tolist()
        elif isinstance(modify_idx, np.ndarray):
            modify_idx = modify_idx.tolist()
        self.benchmark.memory_indices_train[task] = modify_idx

    def get_memory_idx(self, task):
        return self.benchmark.memory_indices_train[task]

    def train_single_step(self, optimizer, criterion, _loader, task, step, n_substeps):
        criterion = nn.CrossEntropyLoss().to(self.device)
        is_last_step = True if step == n_substeps else False # last task
        rs = np.random.RandomState(0)
        # if config['select_type'] in coreset_methods:
        if False:
            pass
            summarizer = Summarizer.factory(self.select_type, rs)

        candidates_indices=[]
        for batch_idx, items in enumerate(_loader):
            item_to_devices = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in items]
            data, target, task_id, indices, *_ = item_to_devices
            if isinstance(data, list):
                data = [x.to(self.device) for x in data]
            data_length = len(target)
            self.backbone.train()
            # data = data.to(self.device)
            # target = target.to(self.device)
            is_rand_start = True if ((step == 1) and (batch_idx < self.r2c_iter) and self.is_r2c) else False
            is_ocspick = True if (self.ocspick and data_length > self.batch_size) else False
            optimizer.zero_grad()
            if is_ocspick and not is_rand_start:

                _eg = compute_and_flatten_example_grads(self.backbone, criterion, data, target, task_id, self.device)
                _g = torch.mean(_eg, 0)
                sorted = sample_selection(_g, _eg, self.tau) # np array
                pick = sorted[:self.batch_size]
                optimizer.zero_grad()
                if isinstance(data, list):
                    picked_data = [x[pick] for x in data]
                else:
                    picked_data = data[pick]
                pred = self.backbone(picked_data, task_id)
                loss = criterion(pred, target[pick])
                loss.backward()

                # Select coresets at final step
                if is_last_step:
                    # candidates_indices.append(pick)
                    candidates_indices.extend(indices[pick].cpu().numpy().tolist()) #

            # elif config['select_type'] in coreset_methods:
            elif False:
                pass
                size = min(data_length, self.batch_size)
                pick = torch.randperm(data_length)[:size]
                if data_length > self.batch_size:
                    selected_pick = summarizer.build_summary(data.cpu().numpy(), target.cpu().numpy(), self.batch_size, method=config['select_type'], model=self.backbone, device=self.device, taskid=task_id)
                pred = self.backbone(data[pick], task_id)
                loss = criterion(pred, target[pick])
                loss.backward()
                if is_last_step:
                    if data_length > self.batch_size:
                        candidates_indices.append(selected_pick)
            else:
                size = min(data_length, self.batch_size)
                pick = torch.randperm(data_length)[:size]
                if isinstance(data, list):
                    picked_data = [x[pick] for x in data]
                else:
                    picked_data = data[pick]

                pred = self.backbone(picked_data, task_id)
                loss = criterion(pred, target[pick])
                loss.backward()
            optimizer.step()

        if is_last_step:
            self.select_coreset(task, candidates_indices)

    def reconstruct_coreset_loader2(self, task):
        trains = []
        all_coreset = {}
        n_classes = self.benchmark.num_classes_per_split
        for tid in range(1,task+1):
            if 'mixture' in self.params['dataset'].lower():
                num_examples_per_task = n_classes[tid]
            else:
                num_examples_per_task = self.memory_size // task
            if not self.bert:
                input_dim = self.backbone.input_dim
                coreset = Coreset(num_examples_per_task, input_shape=[input_dim])
            else:
                input_dim = self.backbone.bert.config.hidden_size
                coreset = Coreset(num_examples_per_task, input_shape=[3, input_dim])

            tid_dataloader = self.benchmark.trains[tid]
            tid_coreset, tid_targets = \
                tid_dataloader.getitem_test_transform_list(self.get_memory_idx(tid))
            tid_coreset = self.data_to_tensor(tid_coreset)
            print(f"reconstruct_coreset_loader2")
            print(f"{tid_coreset.shape=}")
            tid_targets = torch.tensor(tid_targets)

            pick_idx = torch.randperm(num_examples_per_task)
            coreset.data = copy.deepcopy(tid_coreset[pick_idx])
            coreset.targets = copy.deepcopy(tid_targets[pick_idx])
            coreset_loader = torch.utils.data.DataLoader(coreset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
            train_loader = fast_cifar_loader(coreset_loader, tid, eval=False)
            # if 'mnist' in self.params['dataset'].lower():
            #     train_loader = fast_mnist_loader(coreset_loader, eval=False)

            trains += train_loader
        all_coreset = random.sample(trains[:], len(trains))
        return all_coreset

    def train_ocs_single_step(self, optimizer, criterion, _loader, task, step, n_substeps):
        criterion = nn.CrossEntropyLoss().to(self.device)
        is_last_step = True if step == n_substeps else False

        prev_coreset, prev_targets = list(), list()
        for tid in range(1, task):
            tid_dataloader = self.benchmark.trains[tid]
            tid_coreset, tid_targets = \
                tid_dataloader.getitem_test_transform_list(self.get_memory_idx(tid))
            tid_coreset = self.data_to_tensor(tid_coreset)
            tid_targets = torch.tensor(tid_targets)
            prev_coreset.append(tid_coreset)
            prev_targets.append(tid_targets)
        # prev_coreset = [loader['coreset'][tid]['train'].data for tid in range(1, task)]
        # prev_targets = [loader['coreset'][tid]['train'].targets for tid in range(1, task)]
        c_x = torch.cat(prev_coreset, 0).to(self.device)
        print(f"{c_x.shape=}")
        c_y = torch.cat(prev_targets, 0).long().to(self.device)

        ref_loader = self.reconstruct_coreset_loader2(task-1)
        ref_iterloader = iter(ref_loader)

        candidates_indices=[]
        for batch_idx, items in enumerate(_loader):
            item_to_devices = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in items]
            data, target, task_id, indices, *_ = item_to_devices
            data_length = len(target)
            self.backbone.eval()
            optimizer.zero_grad()
            is_rand_start = True if ((step == 1) and (batch_idx < self.r2c_iter) and self.is_r2c) else False
            # Compute reference grads
            ref_pred = self.backbone(c_x, task)
            ref_loss = criterion(ref_pred, c_y)
            ref_loss.backward()
            ref_grads = copy.deepcopy(flatten_grads(self.backbone))
            optimizer.zero_grad()

            data = data.to(self.device)
            target = target.to(self.device)
            if is_rand_start:
                size = min(data_length, self.batch_size)
                pick = torch.randperm(data_length)[:size]
            else:
                # Coreset update
                _eg = compute_and_flatten_example_grads(self.backbone, criterion, data, target, task_id, self.device)
                _g = torch.mean(_eg, 0)
                sorted = sample_selection(_g.to(self.device), _eg.to(self.device), self.tau, ref_grads=ref_grads)
                pick = sorted[:self.batch_size]

            self.backbone.train()
            optimizer.zero_grad()
            pred = self.backbone(data[pick], task_id)
            loss = criterion(pred, target[pick])

            try:
                ref_data = next(ref_iterloader)
            except StopIteration:
                ref_iterloader = iter(ref_loader)
                ref_data = next(ref_iterloader)

            ref_loss = get_coreset_loss(self.backbone, ref_data, self.device)
            loss += self.ref_hyp * ref_loss
            loss.backward()
            optimizer.step()

            if is_last_step:
                # candidates_indices.append(pick)
                candidates_indices.extend(indices[pick].cpu().numpy().tolist()) # 
                print(f"train_ocs_single_step {len(candidates_indices)=}")

        if is_last_step:
            self.select_coreset(task, candidates_indices)
            self.update_coreset(task, task_id)

    def select_coreset(self, task, candidates, candidate_size=1000, fair_selection=True):
        """
        difference from original code
            * candidates from original code is index from each bach
            * now it is index from dataset
        """
        criterion = nn.CrossEntropyLoss().to(self.device)
        n_classes = self.benchmark.num_classes_per_split

        temp_optimizer = self.prepare_optimizer(task)
        temp_optimizer.zero_grad()

        if fair_selection:
            # collect candidates
            cand_data, cand_target = [], []
            current_dataloader = self.benchmark.trains[task]
            cand_data, cand_target = current_dataloader.getitem_test_transform_list(candidates)

            if not self.bert:
                if isinstance(cand_data[0], np.ndarray):
                    cand_data = [torch.from_numpy(cand) for cand in cand_data]
                cand_data = torch.stack(cand_data, 0)
            else:
                if isinstance(cand_data[0][0], np.ndarray):
                    cand_data = [[torch.from_numpy(cand) for cand in x] for x in cand_data]
                cand_data = [torch.stack(x, 0) for x in cand_data]
                cand_data = torch.stack(cand_data, 0)
            cand_target = torch.tensor(cand_target)
            random_pick_up = torch.randperm(len(cand_target))[:candidate_size]

            cand_data = cand_data[random_pick_up]
            cand_target = cand_target[random_pick_up]

            # only works for non-random class idx
            num_per_label = [len((cand_target==(jj+n_classes*(task-1))).nonzero()) for jj in range(n_classes)]
            #print('num samples per label', num_per_label)

            num_examples_per_task = self.memory_size // task

            # if config['select_type'] in coreset_methods:
            if False:
                pass
                rs = np.random.RandomState(0)
                summarizer = Summarizer.factory(config['select_type'], rs)
                pick = summarizer.build_summary(cand_data.cpu().numpy(), cand_target.cpu().numpy(), num_examples_per_task, method=config['select_type'], model=self.backbone, device=self.device, taskid=loader['sequential'][task]['train'][0][2])
                loader['coreset'][task]['train'].data = copy.deepcopy(cand_data[pick])
                loader['coreset'][task]['train'].targets = copy.deepcopy(cand_target[pick])
            else:
                fast_dataset = FastDataset(cand_data, cand_target)
                fast_loader = torch.utils.data.DataLoader(fast_dataset, batch_size=self.params['batch_size_memory'], shuffle=False)
                for x, y in fast_loader:
                    pred = self.backbone(x.to(self.device))
                    loss = criterion(pred, y.long().to(self.device))
                    loss.backward(retain_graph=True)                

                # Coreset update
                _eg = compute_and_flatten_example_grads(self.backbone, criterion, cand_data, cand_target, task, self.device)
                _g = torch.mean(_eg, 0)
                sorted = sample_selection(_g, _eg, self.tau)

                pick = torch.randperm(len(sorted))
                selected = self.classwise_fair_selection(task, cand_target, pick, num_per_label, is_shuffle=True)
                if not isinstance(candidates, list):
                    print(f"{type(candidates)=}")
                
                candidates = np.array(candidates)
                overall_selected_idx = candidates[random_pick_up][selected]
                self.modify_memory_idx(task, overall_selected_idx)
                num_per_label = [len((cand_target[selected]==(jj+n_classes*(task-1))).nonzero()) for jj in range(n_classes)]
                print('after select_coreset, num samples per label', num_per_label)
        else:
            pass

    def update_coreset(self, task, task_id):
        # Coreset update
        num_examples_per_task = self.memory_size // task
        prv_nept = self.memory_size // (task-1)
        n_classes = self.benchmark.num_classes_per_split

        for tid in range(1, task):
            if False:
            # if config['select_type'] in coreset_methods:
                pass
                xx = num_examples_per_task if tid == 1 else prv_nept
                tid_coreset = loader['coreset'][tid]['train'].data
                tid_targets = loader['coreset'][tid]['train'].targets
                class_idx = [tid_targets.cpu().numpy() == i for i in range(config['n_classes'])]
                num_per_label = [len((tid_targets.cpu()==jj).nonzero()) for jj in range(config['n_classes'])]
                rs = np.random.RandomState(0)
                summarizer = Summarizer.factory(config['select_type'], rs)
                selected = summarizer.build_summary(loader['coreset'][tid]['train'].data.cpu().numpy(), loader['coreset'][tid]['train'].targets.cpu().numpy(), num_examples_per_task, method=config['select_type'], model=self.backbone, device=self.device, taskid=tid)
            elif True:
            # elif config['select_type'] == 'ocs_select':
                criterion = nn.CrossEntropyLoss().to(self.device)
                # temp_optimizer = torch.optim.SGD(self.backbone.parameters(), lr=config['seq_lr'], momentum=config['momentum'])
                temp_optimizer = self.prepare_optimizer(tid)
                tid_dataloader = self.benchmark.trains[tid]
                tid_coreset, tid_targets = \
                    tid_dataloader.getitem_test_transform_list(self.get_memory_idx(tid))
                
                tid_coreset = self.data_to_tensor(tid_coreset)
                tid_coreset = tid_coreset.to(self.device)
                tid_targets = torch.tensor(tid_targets)
                temp_optimizer.zero_grad()

                pred = self.backbone(tid_coreset)
                # pred = self.backbone(tid_coreset.to(self.device), task_id)
                loss = criterion(pred, tid_targets.long().to(self.device))
                loss.backward()
                _tid_eg = compute_and_flatten_example_grads(self.backbone, criterion, tid_coreset, tid_targets.to(self.device), tid, self.device)
                _tid_g = torch.mean(_tid_eg, 0)
                pick = sample_selection(_tid_g, _tid_eg, self.tau)

                class_idx = [tid_targets.cpu().numpy() == i for i in range(n_classes)]
                num_per_label = [len((tid_targets.cpu()==(jj+n_classes*(task-1))).nonzero()) for jj in range(n_classes)]
                print('during update_coreset, num samples per label', num_per_label)

                selected = self.classwise_fair_selection(task, tid_targets, pick, num_per_label)
            _nn = [len((tid_targets[selected]==(jj+n_classes*(tid-1))).nonzero()) for jj in range(n_classes)]

            overall_selected_idx = np.array(self.get_memory_idx(tid))[selected]
            self.modify_memory_idx(tid, overall_selected_idx)
            # loader['coreset'][tid]['train'].data = copy.deepcopy(loader['coreset'][tid]['train'].data[selected])
            # loader['coreset'][tid]['train'].targets = copy.deepcopy(loader['coreset'][tid]['train'].targets[selected])

    def classwise_fair_selection(self, task, cand_target, sorted_index, num_per_label, is_shuffle=True):
        num_examples_per_task = self.memory_size // task
        # if num_examples_per_task > len(cand_target):
        #     num_examples_per_task = len(cand_target)
        n_classes = self.benchmark.num_classes_per_split
        num_examples_per_class = num_examples_per_task // n_classes
        num_residuals = num_examples_per_task - num_examples_per_class * n_classes
        residuals =  np.sum([(num_examples_per_class - n_c)*(num_examples_per_class > n_c) for n_c in num_per_label])
        num_residuals += residuals

        n_less_sample_class =  np.sum([(num_examples_per_class > n_c) for n_c in num_per_label])

        # Get the number of coreset instances per class
        while True:
            n_less_sample_class =  np.sum([(num_examples_per_class > n_c) for n_c in num_per_label])
            num_class = (n_classes-n_less_sample_class)
            if num_class == 0:
                break
            elif (num_residuals // num_class) > 0:
                num_examples_per_class += (num_residuals // num_class)
                num_residuals -= (num_residuals // num_class) * num_class
            else:
                break
        # Get best coresets per class
        selected = []
        target_tid = np.floor(max(cand_target)/n_classes)

        for j in range(n_classes):
            position = np.squeeze((cand_target[sorted_index]==j+(target_tid*n_classes)).nonzero())
            if position.numel() > 1:
                selected.append(position[:num_examples_per_class])
            elif position.numel() == 0:
                continue
            else:
                selected.append([position])
        # Fill rest space as best residuals
        selected = np.concatenate(selected)
        print(f"{selected=}")
        unselected = np.array(list(set(np.arange(num_examples_per_task))^set(selected)))
        final_num_residuals = num_examples_per_task - len(selected)
        if final_num_residuals > 0:
            best_residuals = unselected[:final_num_residuals]
            print(f"{final_num_residuals=}")
            print(f"{unselected=}")
            selected = np.concatenate([selected, best_residuals])

        if is_shuffle: 
            random.shuffle(selected)

        return sorted_index[selected.astype(int)]

# return idx of the metric(?)
def sample_selection(g, eg, tau, ref_grads=None, attn=None):

    ng = torch.norm(g)
    neg = torch.norm(eg, dim=1)
    mean_sim = torch.matmul(g,eg.t()) / torch.maximum(ng*neg, torch.ones_like(neg)*1e-6)

    negd = torch.unsqueeze(neg, 1)

    cross_div = torch.matmul(eg,eg.t()) / torch.maximum(torch.matmul(negd, negd.t()), torch.ones_like(negd)*1e-6)
    mean_div = torch.mean(cross_div, 0)

    coreset_aff = 0.
    if ref_grads is not None:
        ref_ng = torch.norm(ref_grads)
        coreset_aff = torch.matmul(ref_grads, eg.t()) / torch.maximum(ref_ng*neg, torch.ones_like(neg)*1e-6)

    measure = mean_sim - mean_div + tau * coreset_aff
    _, u_idx = torch.sort(measure, descending=True)
    return u_idx.cpu().numpy()

def compute_and_flatten_example_grads(m, criterion, data, target, task_id, device):
    _eg = []
    criterion2 = nn.CrossEntropyLoss(reduction='none').to(device)
    m.eval()
    m.zero_grad()

    fast_dataset = FastDataset(data, target)
    fast_loader = torch.utils.data.DataLoader(fast_dataset, batch_size=64, shuffle=False)
    for x, y in fast_loader:
        data_len = len(y)
        if isinstance(data, list):
            x = [xx.to(device) for xx in x]
        else:
            x = x.to(device)
        y = y.to(device)
        pred = m(x)
        loss = criterion2(pred, y)

        for idx in range(data_len):
            loss[idx].backward(retain_graph=True)
            _g = flatten_grads(m, numpy_output=True)
            _eg.append(torch.Tensor(_g))
            m.zero_grad

    return torch.stack(_eg)

def flatten_grads(m, numpy_output=False, bias=True, only_linear=False):
    total_grads = []
    for name, param in m.named_parameters():
        if only_linear:
            if (bias or not 'bias' in name) and 'linear' in name:
                total_grads.append(param.grad.detach().view(-1))
        else:
            if (bias or not 'bias' in name) and not 'bn' in name and not 'IC' in name:
                try:
                    total_grads.append(param.grad.detach().view(-1))
                except AttributeError:
                    pass
                    #print('no_grad', name)
    total_grads = torch.cat(total_grads)
    if numpy_output:
        return total_grads.cpu().detach().numpy()
    return total_grads

def get_coreset_loss(net, iterloader, device):
    criterion = nn.CrossEntropyLoss().to(device)
    net.train()
    coreset_loss = 0
    count = 0
    data, target, task_id = iterloader
    count += len(target)
    data = data.to(device)
    target = target.to(device)
    output = net(data, task_id)
    coreset_loss += criterion(output, target.long())
    coreset_loss /= count
    return coreset_loss