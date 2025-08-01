import torch
import numpy as np
from typing import Dict, Iterable, Optional
import cl_gym as cl
from .base import ContinualTrainer1
from .fair_trainer import FairContinualTrainer2

class ImbalanceContinualTrainer1(FairContinualTrainer2):
    # this is for std, EER fairness metrics
    def validate_algorithm_on_task(self, *args, **kwargs):
        return ContinualTrainer1.validate_algorithm_on_task(self, *args, **kwargs)