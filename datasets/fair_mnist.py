import torchvision
from typing import Any, Callable, Tuple, Optional, Dict, List
from cl_gym.benchmarks.utils import DEFAULT_DATASET_DIR
from cl_gym.benchmarks.transforms import get_default_mnist_transform
from cl_gym.benchmarks.transforms import get_default_rotation_mnist_transform
from cl_gym.benchmarks.transforms import get_default_permuted_mnist_transform
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


class FairMNIST(torchvision.datasets.MNIST):
    def __init__(self, root: str, 
                 train: bool = True, 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None, 
                 download: bool = False, 
                 sensitive_idx: np.array = None,
                 ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.sensitive_idx = sensitive_idx
        if self.sensitive_idx is not None:
            self.add_sensitive_idx(sensitive_idx)

    def add_sensitive_idx(self, sensitive_idx: np.array = None,):
        self.sensitive_idx = sensitive_idx
        if self.sensitive_idx is not None:
            self.sensitive = np.zeros_like(self.targets)
            self.sensitive[self.sensitive_idx] = np.ones_like(self.sensitive_idx)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.sensitive_idx is None:
            return super().__getitem__(index)
        else:
            img, target = super().__getitem__(index)
            return img, target, self.sensitive[index]
            # return super().__getitem__(index), self.sensitive[index]

class FairSplitDataset(SplitDataset):
    def __init__(self, task_id, num_classes_per_split, dataset, class_idx = None):
        self.inputs = []
        self.targets = []
        self.sensitives = [] #ADDED
        self.task_id = task_id
        self.num_classes_per_split = num_classes_per_split
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

    def __build_split(self, dataset, task_id):
        start_class = (task_id-1) * self.num_classes_per_split
        end_class = task_id * self.num_classes_per_split
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
        for idx in selected_indices:
            # img, target = dataset[idx]
            img, target, sensitive = dataset[idx]
            target = torch.tensor(target)
            self.inputs.append(img)
            self.targets.append(target)
            self.sensitives.append(sensitive)
        
        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)

    def __getitem__(self, index: int):
        img, target = self.inputs[index], int(self.targets[index])
        sensitive = self.sensitives[index]
        return img, target, self.task_id, sensitive


class NoiseMNIST(SplitMNIST):
    """
    Split MNIST benchmark.
    The benchmark can have at most 5 tasks, each a binary classification on MNIST digits.
    """
    def __init__(self,
                 num_tasks: int,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_input_transforms: Optional[list] = None,
                 task_target_transforms: Optional[list] = None,
                 noise_size = 0.1,
                 random_class_idx=False):
        # seed = SEED
        # np.random.seed(SEED)
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False

        if num_tasks > 5:
            raise ValueError("Split MNIST benchmark can have at most 5 tasks (i.e., 10 classes, 2 per task)")
        if task_input_transforms is None:
            task_input_transforms = get_default_mnist_transform(num_tasks)
        self.noise_size = noise_size

        cls = np.arange(10)
        if random_class_idx:
            self.class_idx = np.random.choice(cls, len(cls), replace=False)
        else:
            self.class_idx = cls
        print(f"{self.class_idx}")

        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples, task_input_transforms, task_target_transforms)


    def __load_mnist(self):
        self.transform = self.task_input_transforms[0]
        self.mnist_train = FairMNIST(DEFAULT_DATASET_DIR, train=True, download=True, transform=self.transform)
        self.mnist_test = FairMNIST(DEFAULT_DATASET_DIR, train=False, download=True, transform=self.transform)
        
        train_len = len(self.mnist_train)
        test_len = len(self.mnist_test)
        train_sensitive_idx = np.sort(np.random.choice(train_len, train_len//2, replace=False))
        test_sensitive_idx = np.sort(np.random.choice(test_len, test_len//2, replace=False))

        self.mnist_train.add_sensitive_idx(train_sensitive_idx)
        self.mnist_test.add_sensitive_idx(test_sensitive_idx)

        sensitive_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation([90, 90]),
            torchvision.transforms.RandomRotation([0, 0]),
            torchvision.transforms.RandomErasing(p = 1, scale=(self.noise_size, self.noise_size)),
        ])
        tranform_on_idx(self.mnist_train.data, train_sensitive_idx, sensitive_transform)
        tranform_on_idx(self.mnist_test.data, test_sensitive_idx, sensitive_transform)

        self.train_sensitive_idx = train_sensitive_idx
        self.test_sensitive_idx = test_sensitive_idx

    def load_datasets(self):
        self.__load_mnist()
        for task in range(1, self.num_tasks + 1):
            self.trains[task] = FairSplitDataset(task, 2, self.mnist_train, class_idx = self.class_idx)
            self.tests[task] = FairSplitDataset(task, 2, self.mnist_test, class_idx = self.class_idx)

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
