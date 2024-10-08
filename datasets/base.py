import numpy as np
import torch
from PIL import Image
from cl_gym.benchmarks.base import SplitDataset

def tranform_on_idx(data, idx, transform):
    # if len(data) != len(idx):
    #     raise ValueError(f"size of data({len(data)}) and index({len(idx)}) is different")
    transformed = transform(data[idx])
    data[idx] = transformed
    return data

class SplitDataset1(SplitDataset):
    def __init__(self, task_id, num_classes_per_split, dataset, class_idx = None):
        self.task_id = task_id
        self.num_classes_per_split = num_classes_per_split
        self.dataset = dataset
        if isinstance(dataset.targets, np.ndarray):
            original_target = dataset.targets
        elif isinstance(dataset.targets, list):
            original_target = np.asarray(dataset.targets)
        # for MNIST-like datasets where targets are tensors
        elif isinstance(dataset.targets, torch.Tensor):
            original_target = dataset.targets.clone().detach().numpy()
        else:
            raise NotImplementedError
        self.original_target = original_target
        self.class_idx = np.unique(original_target) if class_idx is None else class_idx
        self.build_split(task_id)
        self.sample_weight = torch.ones(self.__len__()) #ADDED - for dtype agreement
    
    def update_weight(self, sample_weight):
        self.sample_weight = sample_weight

    def build_split(self, task_id):
        target_classes = list()
        if isinstance(task_id, int):
            task_id = [task_id]
        for task in task_id:
            start_class = (task-1) * self.num_classes_per_split
            end_class = min(task * self.num_classes_per_split, len(self.class_idx))
            target_classes.extend(self.class_idx[start_class:end_class])
        indices = np.zeros_like(self.original_target)
        for c in target_classes:
            indices = np.logical_or(indices, self.original_target == c)
        self.true_index = np.where(indices)[0] 
        self.targets = self.original_target[self.true_index]

    def __getitem__(self, index: int):
        idx = self.true_index[index]
        img, target, *_ = self.dataset[idx]
        target_ = target if not isinstance(target, torch.Tensor) else target.item()
        if isinstance(self.task_id, int):
            task_id = self.task_id
        else:
            task_id = np.where(self.class_idx == target_)[0][0] // self.num_classes_per_split + 1
        sample_weight = self.sample_weight[index]
        return img, target, task_id, index, sample_weight

    def getitem_test_transform(self, index: int, true_index=False):
        if true_index:
            idx = index
        else:
            idx = self.true_index[index]
        img = self.dataset.data[idx]
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        target = int(self.dataset.targets[idx])
        mode = "RGB" # TODO: need to modify mode for each dataset
        if self.dataset.data.shape[-1] == 3:
            mode = "RGB"
        else:
            mode = "L"
        if hasattr(self.dataset, "test_transform"):
            img = Image.fromarray(img, mode=mode) 
            img = self.dataset.test_transform(img)
        elif hasattr(self.dataset, "transform"):
            img = Image.fromarray(img, mode=mode) 
            img = self.dataset.transform(img)
        else: # Non-image dataset
            pass
        return img, target

    def getitem_test_transform_list(self, indices: list, true_index=False):
        img_list, target_list = [], []
        for idx in indices:
            img, target = self.getitem_test_transform(idx, true_index=true_index)
            img_list.append(img)
            target_list.append(target)
        return img_list, target_list

    def __len__(self):
        return len(self.true_index)

    def __clear_dataset(self):
        original_shape = self.dataset.data.shape
        self.dataset.data = torch.zeros([0, *original_shape[1:]], dtype=torch.uint8)
        self.dataset.targets = torch.zeros([0], dtype=int)
        self.original_target = np.zeros([0], dtype=int)
        self.targets = np.zeros([0], dtype=int)
        self.true_index = np.zeros([0], dtype=int)
        self.sample_weight = torch.ones([0])

    def __add_data(self, X, y, weight=None):
        # only work for dataset (uint image)
        if not X.shape[0] == y.shape[0]:
            raise ValueError(f"Wrong size: {X.shape=}, {y.shape=}")
        if weight is None:
            weight = torch.ones_like(y)
        else:
            if not weight.shape == y.shape:
                ValueError(f"Wrong size: {X.shape=}, {y.shape=}, {weight.shape=}") 
        original_dataset_len = len(self.dataset)
        self.dataset.data = torch.cat([self.dataset.data, X], dim=0)
        self.dataset.targets = torch.cat([self.dataset.targets, y], dim=0)
        self.targets = np.concatenate([self.targets, y], axis=0)
        append_idx = np.arange(original_dataset_len, original_dataset_len+len(y), dtype=int)
        self.true_index = np.concatenate([self.true_index, append_idx], axis=0)
        self.sample_weight = torch.cat([self.sample_weight, weight], dim=0)

    def __replace_data(self, X, y, idx, weight=None):
        if X.shape[0] == y.shape[0]:
            raise ValueError(f"Wrong size: {X.shape=}, {y.shape=}")
        if weight is None:
            weight = torch.ones_like(y)
        else:
            if not weight.shape == y.shape:
                ValueError(f"Wrong size: {X.shape=}, {y.shape=}, {weight.shape=}") 
        self.dataset.data[idx] = X.clone()
        self.dataset.targets[idx] = y.clone()
        self.targets[idx] = y.clone().cpu().numpy()
        self.sample_weight[idx] = weight.clone()

class SplitDataset2(SplitDataset1): # For EER datasets
    def __getitem__(self, index: int):
        img, target, task_id, idx, sample_weight = super().__getitem__(index)
        return img, target, task_id, index, sample_weight, target # target as sensitive attribute

class SplitDataset3(SplitDataset1): # For EO, DP datasets
    def __init__(self, task_id, num_classes_per_split, dataset, class_idx = None):
        super().__init__(task_id, num_classes_per_split, dataset, class_idx = class_idx)

    def build_split(self, task_id):
        super().build_split(task_id)
        self.sensitives = self.dataset.sensitives[self.true_index]

    def __getitem__(self, index: int):
        img, target, task_id, idx, sample_weight = super().__getitem__(index)
        sen = int(self.sensitives[index])
        return img, target, task_id, index, sample_weight, sen

class SplitDataset4(SplitDataset3): # For BiasedMNIST
    def __getitem__(self, index: int):
        idx = self.true_index[index]
        img = self.dataset.data[idx]
        target = int(self.dataset.targets[idx])
        sample_weight = self.sample_weight[index]
        sen = int(self.sensitives[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="RGB")

        if self.dataset.transform is not None:
            img = self.dataset.transform(img)

        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        
        target_ = target if not isinstance(target, torch.Tensor) else target.item()
        if isinstance(self.task_id, int):
            task_id = self.task_id
        else:
            task_id = np.where(self.class_idx == target_)[0][0] // self.num_classes_per_split + 1

        return img, target, task_id, index, sample_weight, sen
