import numpy as np

from .cplex_solver import minimax_LP_solver_v2, minsum_LP_solver_v2
from .scipy_solver import minimax_LP_solver_v3, minsum_LP_solver_v3

from .cplex_solver import LS_solver_v2
from .scipy_solver import LS_solver_v3

def LS_to_dual(A: np.array, b: np.array, binary = False):
    if binary:
        raise NotImplementedError

def minimax_LP_solver(*args, **kwargs):
    return minimax_LP_solver_v3(*args, **kwargs)

def minsum_LP_solver(*args, **kwargs):
    return minsum_LP_solver_v3(*args, **kwargs)

def LS_solver(*args, **kwargs):
    return LS_solver_v3(*args, **kwargs)