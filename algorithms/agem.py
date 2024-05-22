import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import AGEM as AGEM_prev
from .baselines import BaselineContinualAlgoritm

class AGEM(AGEM_prev, BaselineContinualAlgoritm):
    def __init__(self, backbone, benchmark, params, requires_memory=True):
        super().__init__(backbone, benchmark, params)