# code adapted from https://github.com/brcsomnath/FaIRL/blob/main/src/dataloader/mnist_data_create.py
# Sustaining Fairness via Incremental Learning, AAAI 2023

import torch
import torchvision
import numpy as np
import pandas as pd
import os
import pickle
import random
from typing import Optional, Tuple, List

from .mnist import MNIST
from .base import SplitDataset1, SplitDataset3, SplitDataset4
from cl_gym.benchmarks.base import Benchmark
from cl_gym.benchmarks.utils import DEFAULT_DATASET_DIR
from ucimlrepo import fetch_ucirepo

class DrugDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, s):
        self.data = X
        self.targets = y
        self.sensitives = s

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.sensitives[idx]


class Drug(Benchmark):
    def __init__(self,
                 num_tasks: int,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_input_transforms: Optional[list] = None,
                 task_target_transforms: Optional[list] = None,
                 joint=False,
                 random_class_idx=False):
        self.joint = joint
        cls = 6
        self.num_classes_per_split = cls // num_tasks

        if random_class_idx:
            self.class_idx = np.random.choice(cls, len(cls), replace=False)
        else:
            self.class_idx = np.arange(cls)
        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples, per_task_subset_examples,
                         task_input_transforms, task_target_transforms)
        self.load_datasets()
        self.prepare_datasets()

    def __load_drug(self):
        # mnist_train = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=True, download=True, transform=self.transform)
        # mnist_test = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=False, download=True, transform=self.transform)

        # self._modify_dataset(mnist_train, 0.95)
        # self._modify_dataset(mnist_test, 0.5) # s0:s1 = 5:5

        # self.mnist_train = mnist_train
        # self.mnist_test = mnist_test
        pickle_path = os.path.join(DEFAULT_DATASET_DIR, "drug.pickle")
        if os.path.exists(pickle_path):
            # print(f"load from {pickle_path}")
            with open(file=pickle_path, mode='rb') as f:
                data_dict = pickle.load(f)
            X = data_dict['X']
            y = data_dict['y']
        else:
            drug_consumption_quantified = fetch_ucirepo(id=373)
            X = drug_consumption_quantified.data.features 
            y = drug_consumption_quantified.data.targets

            X = X.to_numpy()
            y = y['cannabis'].to_numpy()
            data_dict = {"X":X, "y":y}
            # print(f"save to {pickle_path}")
            with open(file=pickle_path, mode='wb') as f:
                pickle.dump(data_dict, f)

        """
        Cannabis is class of cannabis consumption. 
        It is output attribute with following distribution of classes.
          Value Class                   Frac    Cases
        * CL0   Never Used              21.91%  413
        * CL1   Used over a Decade Ago  10.98%  207
        * CL2   Used in Last Decade     14.11%  266
        * CL3   Used in Last YearAgo    11.19%  211
        * CL4   Used in Last Month      7.43%   140
        * CL5   Used in Last Week       9.81%   185
        * CL6   Used in Last Day        24.56%  463
        Merge CL4 and CL5
        """
        labels = {'CL0':0, 'CL1':1, 'CL2':2, 'CL3':3, 'CL4':4, 'CL5':4, 'CL6':5}
        for i in range(len(y)):
            y[i] = labels[y[i]]

        # split train/test with 70/30
        all_index = np.arange(len(y))
        train_index = np.random.choice(all_index, size=int(len(all_index)*0.7), replace=False)
        test_index = list(set(all_index) - set(train_index))

        train_data = X[train_index]
        train_targets = y[train_index]

        train_data = torch.from_numpy(train_data.astype(np.float32))
        train_targets = torch.from_numpy(train_targets.astype(np.int64))

        test_data = X[test_index]
        test_targets = y[test_index]

        test_data = torch.from_numpy(test_data.astype(np.float32))
        test_targets = torch.from_numpy(test_targets.astype(np.int64))

        train_sens = []
        for i in range(len(train_data)):
            if train_data[i][1] == -0.48246:
                train_sens.append(0)
            elif train_data[i][1] == 0.48246:
                train_sens.append(1)    
        train_sens = np.array(train_sens)

        test_sens = []
        for i in range(len(test_data)):
            if test_data[i][1] == -0.48246:
                test_sens.append(0)
            elif test_data[i][1] == 0.48246:
                test_sens.append(1)
        test_sens = np.array(test_sens)

        self.drug_train = DrugDataset(train_data, train_targets, train_sens)
        self.drug_test = DrugDataset(test_data, test_targets, test_sens)
        self._calculate_yz_num(self.drug_train)

    def _calculate_yz_num(self, dataset):
        sen = dataset.sensitives
        targ = dataset.targets
        m_dict = {s:[0 for _ in self.class_idx] for s in np.unique(sen)}
        for i, e in enumerate(sen):
            m_dict[e][targ[i]]+=1

        # key is sen
        self.m_dict = m_dict 

    def load_datasets(self):
        self.__load_drug()
        for task in range(1, self.num_tasks + 1):
            train_task = task
            if self.joint:
                train_task = [t for t in range(1, task+1)]
            self.trains[task] = SplitDataset3(train_task, self.num_classes_per_split, self.drug_train, class_idx=self.class_idx)
            self.tests[task] = SplitDataset3(task, self.num_classes_per_split, self.drug_test, class_idx=self.class_idx)

    def sample_fair_uniform_class_indices(self, dataset, start_class_idx, end_class_idx, num_samples) -> List:
        sen_rate = 0.5
        num_sens = len(np.unique(dataset.sensitives))
        num_classes = len(self.class_idx)
        target_classes = dataset.targets
        sensitives = dataset.sensitives
        num_examples_per_class = self._calculate_num_examples_per_class(start_class_idx, end_class_idx, num_samples)

        class_indices = []
        for i, cls_idx in enumerate(range(start_class_idx, end_class_idx+1)):
            cls_number = self.class_idx[cls_idx]
            target = (target_classes == cls_number)
            num_g = int(sen_rate * num_examples_per_class[i])
            num_sen_per_class = [num_g, num_examples_per_class[i] - num_g]
            if np.random.random() > 0.5:
                num_sen_per_class[0], num_sen_per_class[1] = num_sen_per_class[1], num_sen_per_class[0]

            # For huge imbalance - lack of s = 1
            avails = list()
            for j in range(num_sens):
                sensitive = (sensitives == j)
                avail = target * sensitive
                num_candidate_examples = len(np.where(avail == 1)[0])
                avails.append(num_candidate_examples)
            diff = [e - num_sen_per_class[k] for k, e in enumerate(avails)]
            for j, e in enumerate(diff):
                if e < 0:
                    while diff[j] < 0 :
                        av = [k > 0 for k in diff]
                        min_value = np.inf
                        min_group = list()
                        for ii, ee in enumerate(num_sen_per_class):
                            if av[ii]:
                                if ee < min_value:
                                    min_group = [ii]
                                    min_value = ee
                                elif ee == min_value:
                                    min_group.append(ii)
                        targ = np.random.choice(min_group, 1)[0]
                        num_sen_per_class[targ] += 1
                        num_sen_per_class[j] -= 1
                        diff = [e - num_sen_per_class[k] for k, e in enumerate(avails)]
                    print(f"class {cls_number}, sen{j} modified")
                    print(f"{num_sen_per_class=}")
                    print(f"{avails=}")

            for j in range(num_sens):
                sensitive = (sensitives == j)
                avail = target * sensitive
                num_candidate_examples = len(np.where(avail == 1)[0])
                if num_candidate_examples < num_sen_per_class[j]:
                    print(f"{num_sen_per_class=}")
                    print(f"{num_candidate_examples=} is too small - smaller than {num_sen_per_class[j]=}")
                    raise AssertionError
                if num_candidate_examples:
                    selected_indices = np.random.choice(np.where(avail == 1)[0],
                                                        num_sen_per_class[j],
                                                        replace=False)
                    class_indices += list(selected_indices)
        return class_indices

    def precompute_memory_indices(self):
        for task in range(1, self.num_tasks + 1):
            start_cls_idx = (task - 1) * 2
            end_cls_idx = task * 2 - 1
            num_examples = self.per_task_memory_examples
            indices_train = self.sample_fair_uniform_class_indices(self.trains[task], start_cls_idx, end_cls_idx, num_examples)
            # indices_train = self.sample_uniform_class_indices(self.trains[task], start_cls_idx, end_cls_idx, num_examples)
            indices_test = self.sample_fair_uniform_class_indices(self.tests[task], start_cls_idx, end_cls_idx, num_examples)
            # indices_test = self.sample_uniform_class_indices(self.tests[task], start_cls_idx, end_cls_idx, num_examples)
            # assert len(indices_train) == len(indices_test) == self.per_task_memory_examples
            assert len(indices_train) == self.per_task_memory_examples
            self.memory_indices_train[task] = indices_train[:]
            self.memory_indices_test[task] = indices_test[:]

    def precompute_seq_indices(self):
        # if self.per_task_seq_examples > len(self.trains[1]):
        #     raise ValueError(f"per task examples = {self.per_task_seq_examples} but first task's examples = {len(self.trains[1])}")
        
        for task in range(1, self.num_tasks+1):
            # self.seq_indices_train[task] = randint(0, len(self.trains[task]), size=self.per_task_seq_examples)
            # self.seq_indices_test[task] = randint(0, len(self.tests[task]), size=min(self.per_task_seq_examples, len(self.tests[task])))
            self.seq_indices_train[task] = sorted(np.random.choice(len(self.trains[task]), size=min(self.per_task_seq_examples, len(self.trains[task])), replace=False).tolist())
            self.seq_indices_test[task] = sorted(np.random.choice(len(self.tests[task]), size=min(self.per_task_seq_examples, len(self.tests[task])), replace=False).tolist())
