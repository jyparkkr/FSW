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


class ResNet2(ResNet_clgym):
    def __init__(self, input_dim: tuple, num_classes: int, multi_head: bool, num_classes_per_head: int, class_idx, block, num_blocks, nf, config: Dict = ...):
        super(ResNet_clgym, self).__init__(multi_head, num_classes_per_head)

        self.in_planes = nf
        self.dim = input_dim[0]
        self.output_dim = num_classes
        self.input_shape = (-1, *input_dim)
        self.num_classes_per_head = num_classes_per_head
        self.class_idx = class_idx

        self.conv1 = conv3x3(self.dim, nf * 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn1 = nn.BatchNorm2d(nf * 1, affine=BN_AFFINE, track_running_stats=False, momentum=BN_MOMENTUM)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, config=config)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, config=config)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, config=config)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, config=config)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def select_output_head(self, *args, **kwargs):
        return select_output_head(*args, **kwargs)

    def forward(self, inp: torch.Tensor, head_ids: Optional[Iterable] = None):
        bsz = inp.size(0)
        shape = (bsz, self.dim, self.input_shape[-2], self.input_shape[-1])

        out = relu(self.bn1(self.conv1(inp.view(shape))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        
        if self.multi_head and head_ids is not None:
            out = self.select_output_head(out, head_ids, self.num_classes_per_head, self.class_idx)
        return out

    def forward_embeds(self, inp: torch.Tensor, head_ids: Optional[Iterable] = None):
        bsz = inp.size(0)
        shape = (bsz, self.dim, self.input_shape[-2], self.input_shape[-1])
        
        out = relu(self.bn1(self.conv1(inp.view(shape))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        embeds = torch.flatten(out, 1)
        out = self.linear(embeds)

        if self.multi_head and head_ids is not None:
            out = self.select_output_head(out, head_ids, self.num_classes_per_head, self.class_idx)
        return out, embeds
    
    def forward_classifier(self, embeds: torch.Tensor, head_ids: Optional[Iterable] = None):
        out = self.linear(embeds)
        if self.multi_head and head_ids is not None:
            out = self.select_output_head(out, head_ids, self.num_classes_per_head, self.class_idx)
        return out
    
class ResNet18Small_clgym(ResNet2):
    def __init__(self, input_dim, output_dim, class_idx=None, multi_head=True, config: dict = {}):
        num_classes_per_head=output_dim//config['num_tasks']
        if class_idx is None:
            class_idx = list(range(output_dim))
            
        super().__init__(input_dim, output_dim, multi_head, num_classes_per_head, class_idx, BasicBlock_clgym, [2, 2, 2, 2], 20, config = config)

class ResNet18_clgym(ResNet2):
    def __init__(self, input_dim, output_dim, class_idx=None, multi_head=True, config: dict = {}):
        num_classes_per_head=output_dim//config['num_tasks']
        if class_idx is None:
            class_idx = list(range(output_dim))
            
        # super().__init__(input_dim, output_dim, multi_head, num_classes_per_head, class_idx, BasicBlock_clgym, [2, 2, 2, 2], 32, config = config)
        super().__init__(input_dim, output_dim, multi_head, num_classes_per_head, class_idx, BasicBlock_clgym, [2, 2, 2, 2], 128, config = config)

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

