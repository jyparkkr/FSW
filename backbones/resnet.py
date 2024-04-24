import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
import random
import cl_gym
from typing import Optional, Dict, Iterable

from cl_gym.backbones.resnet import ResNet, BasicBlock, conv3x3, BN_AFFINE, BN_MOMENTUM
from .base import select_output_head

class ResNet2(ResNet):
    def __init__(self, input_dim: tuple, num_classes: int, multi_head: bool, num_classes_per_head: int, class_idx, block, num_blocks, nf, config: Dict = ...):
        super(ResNet, self).__init__(multi_head, num_classes_per_head)

        self.in_planes = nf
        self.dim = input_dim[0]
        self.output_dim = num_classes
        self.input_shape = (-1, *input_dim)
        self.num_classes_per_head = num_classes_per_head
        self.class_idx = class_idx

        self.conv1 = conv3x3(self.dim, nf * 1)
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
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
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
        out = avg_pool2d(out, 4)
        embeds = out.view(out.size(0), -1)
        out = self.linear(embeds)

        if self.multi_head and head_ids is not None:
            out = self.select_output_head(out, head_ids, self.num_classes_per_head, self.class_idx)
        return out, embeds
    
    def forward_classifier(self, embeds: torch.Tensor, head_ids: Optional[Iterable] = None):
        out = self.linear(embeds)
        if self.multi_head and head_ids is not None:
            out = self.select_output_head(out, head_ids, self.num_classes_per_head, self.class_idx)
        return out
    
class ResNet18Small2(ResNet2):
    def __init__(self, input_dim, output_dim, class_idx=None, multi_head=True, config: dict = {}):
        num_classes_per_head=output_dim//config['num_tasks']
        if class_idx is None:
            class_idx = list(range(output_dim))
            
        super().__init__(input_dim, output_dim, multi_head, num_classes_per_head, class_idx, BasicBlock, [2, 2, 2, 2], 20, config = config)

class ResNet18(ResNet2):
    def __init__(self, input_dim, output_dim, class_idx=None, multi_head=True, config: dict = {}):
        num_classes_per_head=output_dim//config['num_tasks']
        if class_idx is None:
            class_idx = list(range(output_dim))
            
        # super().__init__(input_dim, output_dim, multi_head, num_classes_per_head, class_idx, BasicBlock, [2, 2, 2, 2], 32, config = config)
        super().__init__(input_dim, output_dim, multi_head, num_classes_per_head, class_idx, BasicBlock, [2, 2, 2, 2], 128, config = config)