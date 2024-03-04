import torchvision
from typing import Any, Callable, Tuple, Optional, Dict, List
from cl_gym.benchmarks.utils import DEFAULT_DATASET_DIR
from cl_gym.benchmarks.transforms import get_default_cifar_transform

from cl_gym.benchmarks.base import Benchmark, DynamicTransformDataset, SplitDataset
from cl_gym.benchmarks.mnist import ContinualMNIST, SplitMNIST
from cl_gym.benchmarks.cifar import SplitCIFAR10, SplitCIFAR100
from .mnist import SplitDataset2
import numpy as np
import torch
from PIL import Image


class CIFAR10(SplitCIFAR10):
    """
    Split CIFAR-10 benchmark.
    has 5 tasks, each with 2 classes of CIFAR-10.
    """
    def __init__(self,
                 num_tasks: int,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_input_transforms: Optional[list] = None,
                 task_target_transforms: Optional[list] = None,
                 random_class_idx=False):
        self.random_class_idx = random_class_idx
        self.num_classes_per_split = 2
        cls = np.arange(10)
        if random_class_idx:
            self.class_idx = np.random.choice(cls, len(cls), replace=False)
        else:
            self.class_idx = cls
        print(f"{self.class_idx}")
        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples, task_input_transforms, task_target_transforms)

    def __load_cifar(self):
        transforms = self.task_input_transforms[0]
        CIFAR_dataset = torchvision.datasets.CIFAR100 if self.is_cifar_100 else torchvision.datasets.CIFAR10
        self.cifar_train = CIFAR_dataset(DEFAULT_DATASET_DIR, train=True, download=True, transform=transforms)
        self.cifar_test = CIFAR_dataset(DEFAULT_DATASET_DIR, train=False, download=True, transform=transforms)

    def load_datasets(self):
        self.__load_cifar()
        for task in range(1, self.num_tasks + 1):
            self.trains[task] = SplitDataset2(task, self.num_classes_per_split, self.cifar_train, class_idx=self.class_idx)
            self.tests[task] = SplitDataset2(task, self.num_classes_per_split, self.cifar_test, class_idx=self.class_idx)

    def update_sample_weight(self, task, sample_weight, idx = None):
        """
        true index: self.seq_indices_train[task] (list)
        """
        if idx is None:
            idx = self.seq_indices_train[task]
        weight = self.trains[task].sample_weight
        weight[idx] = sample_weight
        # print(f"{weight=}")
        # print(f"{weight.dtype=}")
        # print(f"{weight.shape=}")
        # print(f"{sample_weight=}")
        # print(f"{sample_weight.dtype=}")
        # print(f"{sample_weight.shape=}")
        self.trains[task].update_weight(weight)

    def precompute_memory_indices(self):
        """
        Precomputes memory indices for each task.
        
        . note::
            The default behavior is class-uniform sampling.
            i.e., each class will have roughly equal number of samples in the memory.
            You can inherit this class and override this method for custom behavior. But a better way
            is to move this logic to your algorithm component's code.
        """
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
        target_classes = dataset.targets.clone().detach().numpy()
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