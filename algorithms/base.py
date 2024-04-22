import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from cl_gym.algorithms import ContinualAlgorithm
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

    def get_num_current_classes(self, task):
        if task is None:
            return self.benchmark.num_classes_per_split
        else:
            if len(self.benchmark.class_idx) - self.benchmark.num_classes_per_split * task < 0:
                return len(self.benchmark.class_idx) - self.benchmark.num_classes_per_split * (task-1)
            else:
                return self.benchmark.num_classes_per_split

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
        num_workers = self.params.get('num_dataloader_workers', torch.get_num_threads())
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

        losses, n_grads_all, n_r_new_grads, new_batch = self.get_loss_grad_all(task_id) 
        # n_grads_all: (class_num) * (weight&bias 차원수)
        # n_r_new_grads: (current step data 후보수) * (weight&bias 차원수)

        print(f"{losses=}")
        # print(f"{n_grads_all.mean(dim=1)=}")

        converter_out = self.converter(losses, self.params['alpha'], n_grads_all, n_r_new_grads, task=task_id)
        optim_in = list()
        for i, e in enumerate(converter_out):
            if i % 2 == 0:
                e_np = e.cpu().detach().numpy().astype('float64')
            else:
                e_np = e.view(-1).cpu().detach().numpy().astype('float64')
            optim_in.append(e_np)

        i = time.time()
        weight = solver(*optim_in)
        
        print(f"Elapsed time:{np.round(time.time()-i, 3)}")
        print(f"Fairness:{np.matmul(optim_in[0], weight)-optim_in[1]}")
        if len(optim_in) >= 4:
            print(f"Current class expected loss:{np.matmul(optim_in[2], weight)-optim_in[3]}")

        tensor_weight = torch.tensor(np.array(weight), dtype=torch.float32)


        self.benchmark.update_sample_weight(task_id, tensor_weight)

        # Need to update self.benchmark.seq_indices_train[task] - to ignore weight = 0
        drop_threshold = 0.05
        updated_seq_indices = np.array(self.benchmark.seq_indices_train[task_id])[np.array(weight)>drop_threshold]
        self.benchmark.seq_indices_train[task_id] = updated_seq_indices.tolist()
        print(f"{len(updated_seq_indices)=}")
        if hasattr(self.benchmark.trains[task_id], "sensitive"):
            print(f"sensitive samples / selected samples = {(self.benchmark.trains[task_id].sensitive[updated_seq_indices] != self.benchmark.trains[task_id].targets[updated_seq_indices]).sum().item()} / {len(updated_seq_indices)}")

        # return self.benchmark.load(task_id, self.params['batch_size_train'],
        #                            num_workers=num_workers, pin_memory=True)[0]

        lists = [list() for _ in new_batch[0]]
        for items in new_batch:
            for i, item in enumerate(items):
                lists[i].append(item)

        args = list()
        for arg in lists:
            cat = torch.cat(arg, dim=0)
            args.append(cat)
        args[4] = tensor_weight

        # for weight figure drawing
        if self.params['dataset'] in ["BiasedMNIST"]:
            sen_weight = dict()
            sen_weight[0] = args[4][args[5]==args[1]].cpu().detach().numpy()
            sen_weight[1] = args[4][args[5]!=args[1]].cpu().detach().numpy()
        else:
            sen_labels = torch.unique(args[5])
            sen_weight = {sen.item():None for sen in sen_labels}
            for k in sen_weight:
                sen_weight[k] = args[4][args[5]==k].cpu().detach().numpy()
        draw_figs(sen_weight, self.params['output_dir'], drop_threshold, \
                  self.params['per_task_examples'], task_id, epoch)
        
        # drop the samples below the threshold
        for i, e in enumerate(args):
            args[i] = e[np.array(weight)>drop_threshold]

        dataset  = WeightModifiedDataset(*args)
        train_loader = DataLoader(dataset, self.params['batch_size_train'], False, num_workers=num_workers,
                                  pin_memory=True)
        
        return train_loader

class WeightModifiedDataset(Dataset):
    def __init__(self, *args):
        self.arglen = len(args)
        self.args = args

    def __len__(self):
        return len(self.args[0])
    
    def __getitem__(self, idx):
        return tuple(arg[idx] for arg in self.args)
    

def draw_figs(weight_dict, output_dir, drop_threshold, y_lim, tid, epoch):
    num_bins = 20
    bins = np.arange(0, 1+1/num_bins, 1/num_bins)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['pdf.fonttype'] = 42
    plt.rc('font', size=15)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=15)
    plt.rc('figure', titlesize=15)
    plt.hist([x for x in weight_dict.values()], bins, stacked=True, \
             edgecolor='black', histtype='bar', label=list(weight_dict.keys()))
    plt.xlim([0, 1])
    plt.ylim([0, y_lim])
    plt.xlabel('Weight')
    plt.ylabel('Number of samples')
    plt.legend(loc='upper center')
    # plt.axvline(drop_threshold, color='black', linestyle='dashed')
    os.makedirs(f"{output_dir}/figs", exist_ok=True)
    plt.savefig(f"{output_dir}/figs/tid_{tid}_epoch_{epoch}_weight_distribution.pdf", bbox_inches="tight")
    # plt.show()
    plt.clf()