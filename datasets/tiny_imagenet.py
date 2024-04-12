import torchvision
import torch
from typing import Any, Callable, Tuple, Optional, Dict, List
from cl_gym.benchmarks.utils import DEFAULT_DATASET_DIR
from cl_gym.benchmarks.base import Benchmark
import numpy as np
 
import torch

from .base import SplitDataset2
from .mnist import MNIST

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # need to check

def get_default_tiny_imagenet_transform(num_tasks: int):
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomResizedCrop(224), # need to check
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return [transforms]*num_tasks

def get_test_tiny_imagenet_transform(num_tasks: int):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return [transforms]*num_tasks


class TinyImageNet(Benchmark):
    """
    Split FashionMNIST benchmark.
    The benchmark can have at most 5 tasks, each a binary classification on Fashion MNIST classes.
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
        self.num_classes_per_split = 10 # depends
        cls = np.arange(100)
        if random_class_idx:
            self.class_idx = np.random.choice(cls, len(cls), replace=False)
        else:
            self.class_idx = cls
        print(f"{self.class_idx}")
        if task_input_transforms is None:
            task_input_transforms = get_default_tiny_imagenet_transform(num_tasks)

        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples, task_input_transforms, task_target_transforms)

    def __load_tiny_imagenet(self):
        transforms = self.task_input_transforms[0]
        # NEED TO IMPLEMENT
        # self.tiny_imagenet_train = torchvision.datasets.FashionMNIST(DEFAULT_DATASET_DIR, train=True, download=True, transform=transforms)
        # self.tiny_imagenet_test = torchvision.datasets.FashionMNIST(DEFAULT_DATASET_DIR, train=False, download=True, transform=transforms)
        
    def load_datasets(self):
        self.__load_tiny_imagenet()
        for task in range(1, self.num_tasks + 1):
            self.trains[task] = SplitDataset2(task, 2, self.fashion_mnist_train, class_idx=self.class_idx)
            self.tests[task] = SplitDataset2(task, 2, self.fashion_mnist_test, class_idx=self.class_idx)