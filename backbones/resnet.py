import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
import random
import cl_gym
from typing import Optional, Dict, Iterable

from cl_gym.backbones.resnet import ResNet as ResNet_clgym, BasicBlock as BasicBlock_clgym, conv3x3, BN_AFFINE, BN_MOMENTUM
from .base import select_output_head

from torchvision.models.resnet import BasicBlock, ResNet

# torchvision + modify(https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html)
class ResNet3(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 is_cifar = False, norm_layer=None, multi_head=False, num_classes_per_head=None, class_idx=None):
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        
        # backbone modification
        if is_cifar:
            print(f"modify resnet for cifar")
            self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.maxpool = nn.Identity()

        self.multi_head: bool = multi_head
        self.num_classes_per_head: int = num_classes_per_head
        if class_idx is None:
            class_idx = list(range(num_classes))
        self.class_idx = class_idx

        if multi_head and num_classes_per_head is None:
            raise ValueError("a Multi-Head Backbone is initiated without defining num_classes_per_head.")

    def select_output_head(self, *args, **kwargs):
        return select_output_head(*args, **kwargs)
    
    def forward(self, x: torch.Tensor, head_ids: Optional[Iterable] = None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.multi_head and head_ids is not None:
            x = self.select_output_head(x, head_ids, self.num_classes_per_head, self.class_idx)
        return x

    def forward_embeds(self, x: torch.Tensor, head_ids: Optional[Iterable] = None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        embeds = torch.flatten(x, 1)
        x = self.fc(embeds)
        if self.multi_head and head_ids is not None:
            x = self.select_output_head(x, head_ids, self.num_classes_per_head, self.class_idx)
        return x, embeds

    def forward_classifier(self, embeds: torch.Tensor, head_ids: Optional[Iterable] = None):
        x = self.fc(embeds)
        if self.multi_head and head_ids is not None:
            x = self.select_output_head(x, head_ids, self.num_classes_per_head, self.class_idx)
        return x

class ResNet18(ResNet3):
    def __init__(self, input_dim, output_dim, class_idx=None, multi_head=True, config: dict = {}):
        num_classes_per_head=output_dim//config['num_tasks']
        if class_idx is None:
            class_idx = list(range(output_dim))
        if "cifar" in config['dataset'].lower():
            is_cifar = True        
        # super().__init__(input_dim, output_dim, multi_head, num_classes_per_head, class_idx, BasicBlock, [2, 2, 2, 2], 32, config = config)
        num_classes_per_head=output_dim//config['num_tasks']
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=output_dim, multi_head=multi_head, \
                         num_classes_per_head=num_classes_per_head, is_cifar=is_cifar, class_idx=class_idx)

class ResNet34(ResNet3):
    def __init__(self, input_dim, output_dim, class_idx=None, multi_head=True, config: dict = {}):
        num_classes_per_head=output_dim//config['num_tasks']
        if class_idx is None:
            class_idx = list(range(output_dim))
        if "cifar" in config['dataset'].lower():
            is_cifar = True
        # super().__init__(input_dim, output_dim, multi_head, num_classes_per_head, class_idx, BasicBlock, [2, 2, 2, 2], 32, config = config)
        num_classes_per_head=output_dim//config['num_tasks']
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=output_dim, multi_head=multi_head, \
                         num_classes_per_head=num_classes_per_head, is_cifar=is_cifar, class_idx=class_idx)

