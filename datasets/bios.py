"""
Need to download *.pickle files first to load bios
You can either download bios file from 
 * https://github.com/Microsoft/biosbias (official repo) 
 * https://github.com/brcsomnath/FaRM/tree/main (containing google drive link)
 * mkdir -p data/biasbios
    wget https://storage.googleapis.com/ai2i/nullspace/biasbios/train.pickle -P data/biasbios/
    wget https://storage.googleapis.com/ai2i/nullspace/biasbios/dev.pickle -P data/biasbios/
    wget https://storage.googleapis.com/ai2i/nullspace/biasbios/test.pickle -P data/biasbios/
Then copy {train, test, dev}.pickle to {DEFAULT_DATASET_DIR}/bios directory
"""

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

from .base import SplitDataset5

class BiosDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids, targets, sensitives):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

        self.targets = targets
        self.sensitives = sensitives
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        data = self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx]
        return data, self.targets[idx], self.sensitives[idx]
    
    def remove_idx(self, indices):
        """
        indices: binary array of whether remove index or not
         * True: remove the index
         * False: remain the index
        """
        remain_idx = np.logical_not(indices)
        self.input_ids = self.input_ids[remain_idx]
        self.attention_mask = self.attention_mask[remain_idx]
        self.token_type_ids = self.token_type_ids[remain_idx]

        self.targets = self.targets[remain_idx]
        self.sensitives = self.sensitives[remain_idx]

class Bios(Benchmark):
    def __init__(self,
                 num_tasks: int,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_input_transforms: Optional[list] = None,
                 task_target_transforms: Optional[list] = None,
                 max_length = 32,
                 joint=False,
                 random_class_idx=False):
        self.num_tasks = num_tasks
        self.num_classes_per_split = 5
        self.joint = joint
        self.add_sensitive_attribute = False
        self.label_names = [
            "accountant", "architect", "attorney", "chiropractor", "comedian",
            "composer", "dentist", "dietitian", "dj", "filmmaker", "interior_designer",
            "journalist", "model", "nurse", "painter", "paralegal", "pastor",
            "personal_trainer", "photographer", "physician", "poet", "professor",
            "psychologist", "rapper", "software_engineer", "surgeon", "teacher",
            "yoga_teacher"
        ]
        self.original_class_idx = np.array([21, 19, 2, 18, 11, 
                                            13, 22, 26, 6, 25, 
                                            1, 14, 12, 20, 9, 
                                            24, 0, 5, 7, 4, 
                                            3, 16, 15, 27, 8, 
                                            10, 17, 23])
        self.class_idx = np.arange(self.num_tasks*self.num_classes_per_split)        
        
        # Bert profile
        self.bert_config = {"tokenizer": "BertTokenizerFast",
                             "model": "bert-base-uncased",
                             "max_length": max_length,
                             "sen": self.add_sensitive_attribute}
        if self.bert_config['tokenizer'] == "BertTokenizerFast":
            Tokenizer = BertTokenizerFast
        else:
            raise NotImplementedError
        
        self.tokenizer = Tokenizer.from_pretrained(self.bert_config['model'])
        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples, task_input_transforms, task_target_transforms)
        self.load_datasets()
        self.prepare_datasets()
    
    def _load(self, filename):
        os.makedirs(filename+".embeddings", exist_ok=True)

        info = [f"{k}={self.bert_config[k]}" for k in self.bert_config]
        info_text = "_".join(info)
        embedding_path = os.path.join(filename+".embeddings", 
                                      info_text+".pickle")
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

            input_ids, attention_mask, token_type_ids = [], [], []
            y_label, g_label = [], []
            label_names_to_num = dict(zip(self.label_names, np.arange(len(self.label_names))))
            c=0
            for row in tqdm(content):
                y = label_names_to_num[row['p']]
                g = 0 if row['g'] == "m" else 1
                txt = row['hard_text_untokenized']
                if self.add_sensitive_attribute:
                    txt = ("MALE. " if g == 0 else "FEMALE. ") + txt

                inputs = self.tokenizer.encode_plus(txt,
                                                     None, 
                                                     max_length = self.bert_config['max_length'],
                                                     padding='max_length',
                                                     truncation=True,
                                                     add_special_tokens=True)
                input_ids.append(inputs['input_ids'])
                attention_mask.append(inputs['attention_mask'])
                token_type_ids.append(inputs['token_type_ids'])
                
                y_label.append(y)
                g_label.append(g)

            dataset = BiosDataset(torch.tensor(input_ids), torch.tensor(attention_mask), 
                                  torch.tensor(token_type_ids), np.array(y_label), np.array(g_label))
            with open(embedding_path, "wb") as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

        # change y order to incoming classes & only remains y in self.class_idx
        m_class_idx = self.original_class_idx[:self.num_tasks*self.num_classes_per_split]
        y_to_class_idx = dict(zip(m_class_idx, np.arange(len(m_class_idx))))
        dataset.targets = np.array([y_to_class_idx.get(y, -1) for y in dataset.targets])
        dataset.remove_idx(dataset.targets < 0)
        return dataset

    def __load_bios(self):
        train_path = f"{DEFAULT_DATASET_DIR}/bios/train.pickle"
        test_path = f"{DEFAULT_DATASET_DIR}/bios/test.pickle"
        dev_path = f"{DEFAULT_DATASET_DIR}/bios/dev.pickle"

        self.bios_train = self._load(train_path)
        self.bios_test = self._load(test_path)
        self.bios_dev = self._load(dev_path)
        self.class_idx = np.arange(len(self.class_idx))
        self._calculate_yz_num(self.bios_train)


    def _calculate_yz_num(self, dataset):
        sen = dataset.sensitives
        targ = dataset.targets
        m_dict = {s:[0 for _ in self.class_idx] for s in np.unique(sen)}
        for i, e in enumerate(sen):
            m_dict[e][targ[i]]+=1

        # key is sen
        self.m_dict = m_dict 

    def load_datasets(self):
        self.__load_bios()
        for task in range(1, self.num_tasks + 1):
            train_task = task
            if self.joint:
                train_task = [t for t in range(1, task+1)]
                print(f"{train_task=}")
            self.trains[task] = SplitDataset5(train_task, self.num_classes_per_split, self.bios_train, class_idx=self.class_idx)
            self.tests[task] = SplitDataset5(task, self.num_classes_per_split, self.bios_test, class_idx=self.class_idx)

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
            indices_train = self.sample_fair_uniform_class_indices(self.trains[task], start_cls_idx, end_cls_idx, num_examples)
            # indices_train = self.sample_uniform_class_indices(self.trains[task], start_cls_idx, end_cls_idx, num_examples)
            indices_test = self.sample_fair_uniform_class_indices(self.tests[task], start_cls_idx, end_cls_idx, num_examples)
            # indices_test = self.sample_uniform_class_indices(self.tests[task], start_cls_idx, end_cls_idx, num_examples)
            # assert len(indices_train) == len(indices_test) == self.per_task_memory_examples
            assert len(indices_train) == self.per_task_memory_examples
            self.memory_indices_train[task] = indices_train[:]
            self.memory_indices_test[task] = indices_test[:]

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
        for task in range(1, self.num_tasks+1):
            self.seq_indices_train[task] = sorted(np.random.choice(len(self.trains[task]), size=min(self.per_task_seq_examples, len(self.trains[task])), replace=False).tolist())
            self.seq_indices_test[task] = sorted(np.random.choice(len(self.tests[task]), size=min(self.per_task_seq_examples, len(self.tests[task])), replace=False).tolist())
