# Need to download *.pkl files first to load bios

import torchvision
import torch
from typing import Any, Callable, Tuple, Optional, Dict, List
from cl_gym.benchmarks.utils import DEFAULT_DATASET_DIR
from cl_gym.benchmarks.base import Benchmark
import numpy as np
from transformers import BertTokenizer, BertTokenizerFast
from tqdm import tqdm

import pickle

import os

from .base import SplitDataset1, SplitDataset3

class BiosDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, sensitives):
        self.data = data
        self.targets = targets
        self.sensitives = sensitives
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.sensitives[idx]


class Bios(Benchmark):
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
        self.num_tasks = num_tasks
        self.num_classes_per_split = 5
        self.joint = joint
        self.label_names = [
            "accountant", "architect", "attorney", "chiropractor", "comedian",
            "composer", "dentist", "dietitian", "dj", "filmmaker", "interior_designer",
            "journalist", "model", "nurse", "painter", "paralegal", "pastor",
            "personal_trainer", "photographer", "physician", "poet", "professor",
            "psychologist", "rapper", "software_engineer", "surgeon", "teacher",
            "yoga_teacher"
        ]
        self.class_idx = np.array([20, 18, 2, 21, 11, 13, 22, 26, 6, 25, 1, 14, 12, 
                                    19, 9, 24, 0, 5, 7, 4, 3, 16, 15, 27, 8, 10, 17, 23])
        print(f"{self.class_idx}")
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-uncased")
        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples, task_input_transforms, task_target_transforms)
        self.load_datasets()
        self.prepare_datasets()
    
    def _load(self, filename):
        os.makedirs(filename+".embeddings", exist_ok=True)
        info = ['BertTokenizerFast', 'bert-base-uncases', 'max_length=32']
        embedding_path = os.path.join(filename+".embeddings", 
                                      ".".join(info)+".pkl")
        if os.path.exists(embedding_path):
            try:
                with open(embedding_path, 'rb') as f:
                    dataset = pickle.load(f)
            except:
                os.remove(embedding_path)
                return self._load(filename)
        else:
            with open(filename, "rb") as f:
                content = pickle.load(f)

            data, y_label, g_label = [], [], []
            for d, txt, y, g in tqdm(content):
                text_emb = [self.tokenizer.encode(txt,
                                                max_length = 32, \
                                                padding='max_length', \
                                                truncation=True, \
                                                add_special_tokens=True)]
                data.append(text_emb[0])
                y_label.append(y)
                g_label.append(g)

            dataset = BiosDataset(torch.tensor(data), np.array(y_label), np.array(g_label))
            with open(embedding_path, "wb") as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

        return dataset

    def __load_bios(self):
        train_path = f"{DEFAULT_DATASET_DIR}/bios/train.pkl"
        test_path = f"{DEFAULT_DATASET_DIR}/bios/test.pkl"
        dev_path = f"{DEFAULT_DATASET_DIR}/bios/dev.pkl"

        self.bios_train = self._load(train_path)
        self.bios_test = self._load(test_path)
        self.bios_dev = self._load(dev_path)

    def load_datasets(self):
        self.__load_bios()
        for task in range(1, self.num_tasks + 1):
            train_task = task
            if self.joint:
                train_task = [t for t in range(1, task+1)]
            self.trains[task] = SplitDataset3(train_task, self.num_classes_per_split, self.bios_train, class_idx=self.class_idx)
            self.tests[task] = SplitDataset3(task, self.num_classes_per_split, self.bios_test, class_idx=self.class_idx)

    def update_sample_weight(self, task, sample_weight, idx = None):
        """
        true index: self.seq_indices_train[task] (list)
        """
        if idx is None:
            idx = self.seq_indices_train[task]
        weight = self.trains[task].sample_weight
        weight[idx] = sample_weight
        self.trains[task].update_weight(weight)

    def precompute_memory_indices(self):
        for task in range(1, self.num_tasks + 1):
            start_cls_idx = (task - 1) * self.num_classes_per_split
            end_cls_idx = task * self.num_classes_per_split - 1
            num_examples = self.per_task_memory_examples
            indices_train = self.sample_uniform_class_indices(self.trains[task], start_cls_idx, end_cls_idx, num_examples)
            indices_test = self.sample_uniform_class_indices(self.tests[task], start_cls_idx, end_cls_idx, num_examples)
            assert len(indices_train) == len(indices_test) == self.per_task_memory_examples
            self.memory_indices_train[task] = indices_train[:]
            self.memory_indices_test[task] = indices_test[:]

    def sample_uniform_class_indices(self, dataset, start_class_idx, end_class_idx, num_samples) -> List:
        target_classes = dataset.targets
        num_examples_per_class = self._calculate_num_examples_per_class(start_class_idx, end_class_idx, num_samples)
        class_indices = []
        # choose num_examples_per_class for each class
        for i, cls_idx in enumerate(range(start_class_idx, end_class_idx+1)):
            cls_number = self.class_idx[cls_idx]
            target = (target_classes == cls_number)
            #  maybe that class doesn't exist
            num_candidate_examples = len(np.where(target == 1)[0])
            if num_candidate_examples:
                selected_indices = np.random.choice(np.where(target == 1)[0],
                                                    min(num_candidate_examples, num_examples_per_class[i]),
                                                    replace=False)
                class_indices += list(selected_indices)
        return class_indices
    
    def precompute_seq_indices(self):
        # if self.per_task_seq_examples > len(self.trains[1]):
        #     raise ValueError(f"per task examples = {self.per_task_seq_examples} but first task's examples = {len(self.trains[1])}")
        
        for task in range(1, self.num_tasks+1):
            # self.seq_indices_train[task] = randint(0, len(self.trains[task]), size=self.per_task_seq_examples)
            # self.seq_indices_test[task] = randint(0, len(self.tests[task]), size=min(self.per_task_seq_examples, len(self.tests[task])))
            self.seq_indices_train[task] = sorted(np.random.choice(len(self.trains[task]), size=min(self.per_task_seq_examples, len(self.trains[task])), replace=False).tolist())
            self.seq_indices_test[task] = sorted(np.random.choice(len(self.tests[task]), size=min(self.per_task_seq_examples, len(self.tests[task])), replace=False).tolist())
