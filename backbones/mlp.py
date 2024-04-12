import torch
import numpy as np
import cl_gym
from typing import Optional, Dict, Iterable

from cl_gym.backbones.mlp import MLP2Layers
from .base import select_output_head

class MLP2Layers2(MLP2Layers):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, class_idx=None, config=None,
                 dropout_prob=0.0, activation='ReLU', bias=True, include_final_layer_act=False):
        if hasattr(input_dim, '__iter__'):
            input_dim = np.prod(input_dim)
        if class_idx is None:
            class_idx = list(range(output_dim))
        self.class_idx = class_idx
        self.num_classes_per_head=output_dim//config['num_tasks']
        super().__init__(num_classes_per_head=self.num_classes_per_head, 
                         input_dim=input_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2,
                         output_dim=output_dim, dropout_prob=dropout_prob, activation=activation,
                         bias=bias, include_final_layer_act=include_final_layer_act)

    def select_output_head(self, *args, **kwargs):
        return select_output_head(*args, **kwargs)

    def forward_embeds(self, inp: torch.Tensor, head_ids: Optional[Iterable] = None):
        inp = inp.view(inp.shape[0], -1)
        out = inp
        for block in self.blocks:
            embeds = out
            out = block(out)
        if self.multi_head:
            embeds = out
            out = self.select_output_head(out, head_ids, self.num_classes_per_head, self.class_idx)
        return out, embeds
    
    def forward_classifier(self, embeds: torch.Tensor, head_ids: Optional[Iterable] = None):
        classifier = self.blocks[-1]
        out = classifier(embeds)
        if self.multi_head:
            out = self.select_output_head(out, head_ids, self.num_classes_per_head, self.class_idx)
        return out