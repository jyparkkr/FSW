import numpy as np

from .cplex_solver import absolute_minimax_LP_solver_v2, absolute_minsum_LP_solver_v2
from .scipy_solver import absolute_minimax_LP_solver_v3, absolute_minsum_LP_solver_v3

from .cplex_solver import LS_solver_v2
from .scipy_solver import LS_solver_v3

from .cplex_solver import absolute_and_nonabsolute_minsum_LP_solver_v1

def LS_to_dual(A: np.array, b: np.array, binary = False):
    if binary:
        raise NotImplementedError

def absolute_minimax_LP_solver(*args, **kwargs):
    return absolute_minimax_LP_solver_v3(*args, **kwargs)

def absolute_minsum_LP_solver(*args, **kwargs):
    return absolute_minsum_LP_solver_v3(*args, **kwargs)

def absolute_and_nonabsolute_minsum_LP_solver(*args, **kwargs):
    return absolute_and_nonabsolute_minsum_LP_solver_v1(*args, **kwargs)

def LS_solver(*args, **kwargs):
    return LS_solver_v3(*args, **kwargs)