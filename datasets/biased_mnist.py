# code adapted from https://github.com/brcsomnath/FaIRL/blob/main/src/dataloader/mnist_data_create.py
# Sustaining Fairness via Incremental Learning, AAAI 2023

# import os

# import numpy as np
# from PIL import Image

# import matplotlib.pyplot as plt

# import torch
# import random
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import grad
# from torchvision import transforms
# from torchvision import datasets
# from collections import defaultdict
# import torchvision.datasets.utils as dataset_utils

# from torchvision.datasets import VisionDataset
# from torch.utils.data import DataLoader, Dataset

# # part of the code adapted
# # from https://arxiv.org/abs/1907.02893

# def add_color_background(background, color):
#     """Adds background color to MNIST digits"""
#     h, w = background.shape
#     r, g, b = color

#     digit_mask = (background[:, :] == 0)
#     digit = np.zeros_like(background)
#     digit[digit_mask] = 255

#     background = np.reshape(background, [h, w, 1])
#     digit = np.reshape(digit, [h, w, 1])
#     arr = np.concatenate([r * background + digit, 
#                             g * background + digit, 
#                             b * background + digit],
#                          axis=2)
#     return arr


# def plot_dataset_digits(dataset):
#     fig = plt.figure(figsize=(25, 10), dpi=400)
#     columns = 5
#     rows = 2
#     # ax enables access to manipulate each of subplots
#     ax = []

#     digit_idx = {}
#     digits = []
#     for i in range(len(dataset)):
#         if len(digit_idx) == 10:
#             break

#         _, l, _ = dataset[i]
#         if l not in digits:
#             digit_idx[l] = (i)
#             digits.append(l)

#     for i in range(columns * rows):
#         img, label, g = dataset[digit_idx[i]]
#         # create subplot and append to ax
#         subplot = fig.add_subplot(rows, columns, i + 1)

#         ax.append(subplot)
#         plt.imshow(img, interpolation='nearest')
#         plt.axis('off')
#     plt.subplots_adjust(wspace=0, hspace=0.1)
#     plt.show()


# path = '../../data/'

# train_mnist = datasets.mnist.MNIST(path, train=True)
# test_mnist = datasets.mnist.MNIST(path, train=False)

# colors = {
#     0: (1, 0, 0),
#     1: (0, 1, 0),
#     2: (1, 1, 0),
#     3: (0, 0, 1),
#     4: (1, 0.65, 0),
#     5: (0.5, 0, 0.5),
#     6: (0, 1, 1),
#     7: (1, 0.75, 0.8),
#     8: (0.8, 1, 0),
#     9: (.588, .294, 0.)
# }


# def add_noise(color, sigma=0.03):
#     return (color[0] + sigma * np.random.standard_normal(),
#             color[1] + sigma * np.random.standard_normal(),
#             color[2] + sigma * np.random.standard_normal())


# train_set = []
# for idx, (im, label) in enumerate(train_mnist):
#     if idx % 10000 == 0:
#         print(f'Converting image {idx}/{len(train_mnist)}')

#     im_array = np.array(im)
#     if np.random.uniform() < 0.9:
#         protected_label = label
#         color = (colors[label])
#     else:
#         protected_label = random.randint(0, 9)
#         color = (colors[protected_label])

#     mask = (im_array[:, :] == 0)
#     background = np.zeros_like(im_array)
#     background[mask] = 255

#     colored_arr = add_color_background(background, color)
#     img = Image.fromarray(colored_arr.astype(np.uint8))
#     train_set.append((img, label, protected_label))

# test_set = []
# for idx, (im, label) in enumerate(test_mnist):
#     if idx % 2000 == 0:
#         print(f'Converting image {idx}/{len(test_mnist)}')

#     im_array = np.array(im)

#     protected_label = random.randint(0, 9)
#     color = colors[protected_label]

#     mask = (im_array[:, :] == 0)
#     background = np.zeros_like(im_array)
#     background[mask] = 255

#     colored_arr = add_color_background(background, color)

#     img = Image.fromarray(colored_arr.astype(np.uint8))
#     test_set.append((img, label, protected_label))

# plot_dataset_digits(train_set)

# colored_mnist_dir = '../../data/colored_mnist-.9/'
# os.makedirs(colored_mnist_dir)

import torch
import torchvision
import numpy as np
from typing import Optional, Tuple, List

from datasets.mnist import MNIST, SplitDataset2, SplitDataset3
from cl_gym.benchmarks.utils import DEFAULT_DATASET_DIR

COLOR_MAP = {
    0: (1, 0, 0),
    1: (0, 1, 0),
    2: (1, 1, 0),
    3: (0, 0, 1),
    4: (1, 0.65, 0),
    5: (0.5, 0, 0.5),
    6: (0, 1, 1),
    7: (1, 0.75, 0.8),
    8: (0.8, 1, 0),
    9: (.588, .294, 0.)
}

from cl_gym.benchmarks.transforms import MNIST_MEAN, MNIST_STD
def get_default_biased_mnist_transform(num_tasks: int):
    r = (1 - 0.1913)
    color_values = np.array(list(COLOR_MAP.values()))
    m_rgb = color_values.mean(axis=0)
    std_rgb = color_values.std(axis=0)

    bmnist_mean = [r*m + MNIST_MEAN[0] for m in m_rgb]
    bmnist_std = [(r*s**2+(1-r)*MNIST_STD[0]**2+r*(1-r)*(bmnist_mean[i] - m_rgb[i])**2)**0.5 for i, s in enumerate(std_rgb)]
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(bmnist_mean, bmnist_std),
    ])
    # return [torchvision.transforms.ToTensor()]*num_tasks
    return [transforms]*num_tasks

class BiasedMNIST(MNIST):
    def __init__(self,
                 num_tasks: int,
                 per_task_examples: Optional[int] = None,
                 per_task_joint_examples: Optional[int] = 0,
                 per_task_memory_examples: Optional[int] = 0,
                 per_task_subset_examples: Optional[int] = 0,
                 task_input_transforms: Optional[list] = None,
                 task_target_transforms: Optional[list] = None,
                 random_class_idx=False):
        if task_input_transforms is None:
            task_input_transforms = get_default_biased_mnist_transform(num_tasks)
        super().__init__(num_tasks, per_task_examples, per_task_joint_examples, per_task_memory_examples,
                         per_task_subset_examples, task_input_transforms, task_target_transforms, random_class_idx=random_class_idx)

    def __load_mnist(self):
        transforms = self.task_input_transforms[0]
        mnist_train = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=True, download=True, transform=transforms)
        mnist_test = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=False, download=True, transform=transforms)

        # self._modify_dataset(mnist_train, 0.9)
        self._modify_dataset(mnist_train, 0.8)
        # self._modify_dataset(mnist_test, 0) # s0:s1 = 1:9
        self._modify_dataset(mnist_test, 4/9) # s0:s1 = 5:5

        self.mnist_train = mnist_train
        self.mnist_test = mnist_test

    def load_datasets(self):
        self.__load_mnist()
        for task in range(1, self.num_tasks + 1):
            self.trains[task] = SplitDataset3(task, self.classes_per_split, self.mnist_train)
            self.tests[task] = SplitDataset3(task, self.classes_per_split, self.mnist_test)

    def _modify_dataset(self, dataset, corr):
        sensitive = torch.zeros_like(dataset.targets)
        labels = dataset.targets

        old = dataset.data
        new = list()
        colors = {k: torch.tensor([round(i*255) for i in v]) for k, v in COLOR_MAP.items()}

        for i, e in enumerate(old):
            if np.random.uniform() < corr:
                sensitive_label = labels[i].item()
            else:
                sensitive_label = np.random.randint(10)
            color = colors[sensitive_label]
            sensitive[i] = sensitive_label
            mask = (e == 0)
            not_mask = (e != 0)
            new.append(mask.unsqueeze(2).repeat(1, 1, 3) * color.reshape(1, 1, -1).repeat(*mask.shape, 1) \
                # +(not_mask * e).unsqueeze(2).repeat(1, 1, 3))
                +(not_mask * 255).unsqueeze(2).repeat(1, 1, 3))
        
        n = torch.stack(new).type(torch.uint8)
        dataset.data = n
        dataset.sensitive = sensitive
    
    def add_noise(color, sigma=0.03):
        return (color[0] + sigma * np.random.standard_normal(),
                color[1] + sigma * np.random.standard_normal(),
                color[2] + sigma * np.random.standard_normal())

    def sample_fair_uniform_class_indices(self, dataset, start_class_idx, end_class_idx, num_samples) -> List:
        sen_rate = 0.5 # 0.5: 반반, 0.1: 모두 똑같게

        num_classes = len(self.class_idx)
        target_classes = dataset.targets.clone().detach().numpy()
        sensitives = dataset.sensitive.clone().detach().numpy()
        num_examples_per_class = self._calculate_num_examples_per_class(start_class_idx, end_class_idx, num_samples)

        class_indices = []
        for i, cls_idx in enumerate(range(start_class_idx, end_class_idx+1)):
            cls_number = self.class_idx[cls_idx]
            num_sen_per_class = [0]*num_classes
            for cls in range(num_classes):
                if cls == cls_number:
                    num_sen_per_class[cls] += int(sen_rate * num_examples_per_class[i])
                else:
                    num_sen_per_class[cls] += int((1-sen_rate)*num_examples_per_class[i]/(num_classes-1))
            # if memory_size can't be divided by num_class classes
            if sum(num_sen_per_class) < num_examples_per_class[i]:
                diff = num_examples_per_class[i] - sum(num_sen_per_class)
                while diff:
                    diff -= 1
                    num_sen_per_class[np.random.randint(0, num_classes)] += 1

            target = (target_classes == cls_number)
            for j in range(num_classes):
                sensitive = (sensitives == j)
                avail = target * sensitive
                num_candidate_examples = len(np.where(avail == 1)[0])
                if num_candidate_examples:
                    selected_indices = np.random.choice(np.where(avail == 1)[0],
                                                        min(num_candidate_examples, num_sen_per_class[j]),
                                                        replace=False)
                    class_indices += list(selected_indices)
        return class_indices

    def precompute_memory_indices(self):
        for task in range(1, self.num_tasks + 1):
            start_cls_idx = (task - 1) * 2
            end_cls_idx = task * 2 - 1
            num_examples = self.per_task_memory_examples
            # indices_train = self.sample_fair_uniform_class_indices(self.trains[task], start_cls_idx, end_cls_idx, num_examples)
            indices_train = self.sample_uniform_class_indices(self.trains[task], start_cls_idx, end_cls_idx, num_examples)
            indices_test = self.sample_fair_uniform_class_indices(self.tests[task], start_cls_idx, end_cls_idx, num_examples)
            # assert len(indices_train) == len(indices_test) == self.per_task_memory_examples
            assert len(indices_train) == self.per_task_memory_examples
            self.memory_indices_train[task] = indices_train[:]
            self.memory_indices_test[task] = indices_test[:]
