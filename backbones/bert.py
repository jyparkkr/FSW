import torch
import torch.nn as nn
import numpy as np
import cl_gym
from typing import Optional, Dict, Iterable
import transformers
from transformers import BertModel
from .base import select_output_head


class BertClassifier(nn.Module):
    def __init__(self, num_classes: int, model_type: str, config: dict):
        super(BertClassifier, self).__init__()
        self.device = config['device']
        self.bert = BertModel.from_pretrained(model_type)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.num_classes_per_head = 5
        self.class_idx = np.arange(25)

    def select_output_head(self, *args, **kwargs):
        return select_output_head(*args, **kwargs)

    def forward_embeds(self, x: torch.Tensor, head_ids: Optional[Iterable] = None):
        if isinstance(x, torch.Tensor):
            x = x.transpose(0, 1)
        output = self.bert.forward(*x)
        embeds = output['pooler_output']
        pooler = self.dropout(embeds)
        x = self.classifier(pooler)
        if head_ids is not None:
            x = self.select_output_head(x, head_ids, self.num_classes_per_head, self.class_idx)
        return x, embeds
    
    def forward(self, *args, **kwargs):
        return self.forward_embeds(*args, **kwargs)[0]

    def forward_classifier(self, embeds: torch.Tensor, head_ids: Optional[Iterable] = None):
        pooler = self.dropout(embeds)
        x = self.classifier(pooler)
        if head_ids is not None:
            x = self.select_output_head(x, head_ids, self.num_classes_per_head, self.class_idx)
        return x