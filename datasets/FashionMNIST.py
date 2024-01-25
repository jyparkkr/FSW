import torchvision
from typing import Any, Callable, Tuple, Optional, Dict, List
from cl_gym.benchmarks.utils import DEFAULT_DATASET_DIR
from cl_gym.benchmarks.transforms import get_default_fashion_mnist_transform

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


class FashionMNIST_modified(torchvision.datasets.FashionMNIST):
    def __init__(self, root: str, 
                 train: bool = True, 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None, 
                 download: bool = False, 
                 ) -> None:
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)
        return img, target


class SplitDataset_modified(SplitDataset):
    def __init__(self, task_id, classes_per_split, dataset, class_idx = None):
        self.inputs = []
        self.targets = []
        self.sample_weight = [] #ADDED
        self.task_id = task_id
        self.classes_per_split = classes_per_split
        if class_idx is None:
            if isinstance(dataset.targets, list):
                target_classes = np.asarray(dataset.targets)
            # for MNIST-like datasets where targets are tensors
            else:
                target_classes = dataset.targets.clone().detach().numpy()
            self.class_idx = np.unique(target_classes)
        else:
            self.class_idx = class_idx
        self.__build_split(dataset, task_id)
        self.sample_weight = torch.ones_like(self.targets)

    
    def update_weight(self, sample_weight):
        self.sample_weight = sample_weight


    def __build_split(self, dataset, task_id):
        start_class = (task_id-1) * self.classes_per_split
        end_class = task_id * self.classes_per_split
        # For CIFAR-like datasets in torchvision where targets are list
        if isinstance(dataset.targets, list):
            target_classes = np.asarray(dataset.targets)
        # for MNIST-like datasets where targets are tensors
        else:
            target_classes = dataset.targets.clone().detach().numpy()
        # target_classes = dataset.targets.clone().detach().numpy()
        indices = np.zeros_like(target_classes)
        for c in self.class_idx[start_class:end_class]:
            indices = np.logical_or(indices, target_classes == c)
        selected_indices = np.where(indices)[0]        
        for i, idx in enumerate(selected_indices):
            img, target = dataset[idx]
            target = torch.tensor(target)
            self.inputs.append(img)
            self.targets.append(target)
        
        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)

    def __getitem__(self, index: int):
        img, target = self.inputs[index], int(self.targets[index])
        sample_weight = self.sample_weight[index]
        return img, target, self.task_id, sample_weight



class FashionMNIST(SplitMNIST):
    def __init__(self,
                 num_tasks: int,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_input_transforms: Optional[list] = None,
                 task_target_transforms: Optional[list] = None,
                 random_class_idx=False,
                 ):
        if num_tasks > 5:
            raise ValueError("Split MNIST benchmark can have at most 5 tasks (i.e., 10 classes, 2 per task)")
        if task_input_transforms is None:
            task_input_transforms = get_default_fashion_mnist_transform(num_tasks)
        self.random_class_idx = random_class_idx

        cls = np.arange(10)
        if random_class_idx:
            self.class_idx = np.random.choice(cls, len(cls), replace=False)
        else:
            self.class_idx = cls
        print(f"{self.class_idx}")

        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples, task_input_transforms, task_target_transforms)


    def update_sample_weight(self, task, sample_weight):
        self.trains[task].update_weight(sample_weight)


    def __load_fashion_mnist(self):
        transforms = self.task_input_transforms[0]
        self.fashion_mnist_train = FashionMNIST_modified(DEFAULT_DATASET_DIR, train=True, download=True, transform=transforms)
        self.fashion_mnist_test = FashionMNIST_modified(DEFAULT_DATASET_DIR, train=False, download=True, transform=transforms)
        

    def load_datasets(self):
        self.__load_fashion_mnist()
        for task in range(1, self.num_tasks + 1):
            self.trains[task] = SplitDataset_modified(task, 2, self.fashion_mnist_train)
            self.tests[task] = SplitDataset_modified(task, 2, self.fashion_mnist_test)


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


    def precompute_memory_indices(self):
        for task in range(1, self.num_tasks + 1):
            start_cls_idx = (task - 1) * 2
            end_cls_idx = task * 2 - 1
            num_examples = self.per_task_memory_examples
            indices_train = self.sample_uniform_class_indices(self.trains[task], start_cls_idx, end_cls_idx, num_examples)
            indices_test = self.sample_uniform_class_indices(self.tests[task], start_cls_idx, end_cls_idx, num_examples)
            assert len(indices_train) == len(indices_test) == self.per_task_memory_examples
            self.memory_indices_train[task] = indices_train[:]
            self.memory_indices_test[task] = indices_test[:]
