import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import AGEM
from .baselines import BaseContinualAlgoritm

class AGEM(AGEM, BaseContinualAlgoritm):
    pass

