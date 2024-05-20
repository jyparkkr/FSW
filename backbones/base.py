import torch
import torch.nn as nn
import numpy as np

from typing import Iterable, Optional, Union, Iterable, Dict


def select_output_head(output, head_ids: Iterable, num_classes_per_split: int, class_idx) -> torch.Tensor:
    """
    Helper method for selecting task-specific head.
    
    Args:
        output: The output of forward-pass. Shape: [BatchSize x ...]
        head_ids: head_ids for each example. Shape [BatchSize]

    Returns:
        output: The output where for each example in batch is calculated from one head in head_ids.
    """
    # TODO: improve performance by vectorizing this operation.
    # TODO: this doesn't work for random index
    # However, not too bad for now since number of classes is small (usually 2 or 5).

    for i, head in enumerate(head_ids):
        offset1 = int((head - 1) * num_classes_per_split)
        offset2 = int(head * num_classes_per_split)
        output[i, class_idx[:offset1]] = -10e10  
        output[i, class_idx[offset2:]] = -10e10  
        return output
