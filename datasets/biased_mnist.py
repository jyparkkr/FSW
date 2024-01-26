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

from datasets.mnist import MNIST
from cl_gym.benchmarks.utils import DEFAULT_DATASET_DIR
from cl_gym.benchmarks.base import Benchmark, DynamicTransformDataset, SplitDataset

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


class BiasedMNIST(MNIST):
    def __load_mnist(self):
        transforms = self.task_input_transforms[0]
        mnist_train = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=True, download=True, transform=transforms)
        mnist_test = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=False, download=True, transform=transforms)

        self._modify_dataset(mnist_train)
        self._modify_dataset(mnist_test)

        self.mnist_train = mnist_train
        self.mnist_test = mnist_test

    def load_datasets(self):
        self.__load_mnist()
        for task in range(1, self.num_tasks + 1):
            self.trains[task] = SplitDataset(task, 2, self.mnist_train)
            self.tests[task] = SplitDataset(task, 2, self.mnist_test)


    def _modify_dataset(self, dataset):
        sensitive_labels = torch.zeros_like(dataset.targets)
        labels = dataset.targets

        old = dataset.data
        new = list()
        colors = {k: torch.tensor(v) for k, v in COLOR_MAP.items()}
        for i, e in enumerate(old):
            if np.random.uniform() < 0.9:
                sensitive = labels[i].item()
            else:
                sensitive = np.random.randint(0, 9)
            color = colors[sensitive]
            sensitive_labels[i] = sensitive

            mask = (e == 0)
            not_mask = (e != 0)
            new.append(mask.unsqueeze(0).repeat(3, 1, 1) * color.reshape(-1, 1, 1).repeat(1, *mask.shape) \
                +not_mask.unsqueeze(0).repeat(3, 1, 1))
        
        n = torch.stack(new)
        dataset.data = n
        dataset.sensitive_labels = sensitive_labels
    

    
    def add_noise(color, sigma=0.03):
        return (color[0] + sigma * np.random.standard_normal(),
                color[1] + sigma * np.random.standard_normal(),
                color[2] + sigma * np.random.standard_normal())



