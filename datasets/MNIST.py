import torchvision
from typing import Any, Callable, Tuple, Optional, Dict, List
from cl_gym.benchmarks.utils import DEFAULT_DATASET_DIR
from cl_gym.benchmarks.transforms import get_default_mnist_transform

from cl_gym.benchmarks.base import Benchmark, DynamicTransformDataset, SplitDataset
from cl_gym.benchmarks.mnist import ContinualMNIST, SplitMNIST
import numpy as np
import torch

def tranform_on_idx(data, idx, transform):
    # if len(data) != len(idx):
    #     raise ValueError(f"size of data({len(data)}) and index({len(idx)}) is different")
    transformed = transform(data[idx])
    data[idx] = transformed
    return data


class MNIST(SplitMNIST):
    def __init__(self,
                 num_tasks: int,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_input_transforms: Optional[list] = None,
                 task_target_transforms: Optional[list] = None,
                 random_class_idx=False):
        if num_tasks > 5:
            raise ValueError("Split MNIST benchmark can have at most 5 tasks (i.e., 10 classes, 2 per task)")
        if task_input_transforms is None:
            task_input_transforms = get_default_mnist_transform(num_tasks)
        self.random_class_idx = random_class_idx

        cls = np.arange(10)
        if random_class_idx:
            self.class_idx = np.random.choice(cls, len(cls), replace=False)
        else:
            self.class_idx = cls
        print(f"{self.class_idx}")

        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples, task_input_transforms, task_target_transforms)


    def __load_mnist(self):
        transforms = self.task_input_transforms[0]
        self.mnist_train = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=True, download=True, transform=transforms)
        self.mnist_test = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=False, download=True, transform=transforms)

    def load_datasets(self):
        self.__load_mnist()
        for task in range(1, self.num_tasks + 1):
            self.trains[task] = SplitDataset(task, 2, self.mnist_train)
            self.tests[task] = SplitDataset(task, 2, self.mnist_test)

    def precompute_memory_indices(self):
        for task in range(1, self.num_tasks + 1):
            start_cls = (task - 1) * 2
            end_cls = task * 2 - 1
            num_examples = self.per_task_memory_examples
            indices_train = self.sample_uniform_class_indices(self.trains[task], start_cls, end_cls, num_examples)
            indices_test = self.sample_uniform_class_indices(self.tests[task], start_cls, end_cls, num_examples)
            assert len(indices_train) == len(indices_test) == self.per_task_memory_examples
            self.memory_indices_train[task] = indices_train[:]
            self.memory_indices_test[task] = indices_test[:]

