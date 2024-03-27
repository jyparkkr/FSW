import torchvision
from typing import Any, Callable, Tuple, Optional, Dict, List
from cl_gym.benchmarks.utils import DEFAULT_DATASET_DIR
from cl_gym.benchmarks.transforms import get_default_fashion_mnist_transform

import numpy as np
import torch

from .base import SplitDataset2
from .mnist import MNIST


class FashionMNIST(MNIST):
    """
    Split FashionMNIST benchmark.
    The benchmark can have at most 5 tasks, each a binary classification on Fashion MNIST classes.
    """
    def __init__(self, 
                 num_tasks: int,
                 task_input_transforms: Optional[list] = None,
                 **kwargs):
        if task_input_transforms is None:
            task_input_transforms = get_default_fashion_mnist_transform(num_tasks)
        super().__init__(num_tasks, task_input_transforms=task_input_transforms, **kwargs)

    def __load_fashion_mnist(self):
        transforms = self.task_input_transforms[0]
        self.fashion_mnist_train = torchvision.datasets.FashionMNIST(DEFAULT_DATASET_DIR, train=True, download=True, transform=transforms)
        self.fashion_mnist_test = torchvision.datasets.FashionMNIST(DEFAULT_DATASET_DIR, train=False, download=True, transform=transforms)
        
    def load_datasets(self):
        self.__load_fashion_mnist()
        for task in range(1, self.num_tasks + 1):
            self.trains[task] = SplitDataset2(task, 2, self.fashion_mnist_train, class_idx=self.class_idx)
            self.tests[task] = SplitDataset2(task, 2, self.fashion_mnist_test, class_idx=self.class_idx)