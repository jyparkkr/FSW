import numpy as np
import torch
from scipy.optimize import linprog, lsq_linear
from .cplex_solver import minimax_LP_solver_v2, minsum_LP_solver_v2


def LS_solver_v3(A: np.ndarray, b: np.ndarray, binary = False):
    soln = lsq_linear(A, b, bounds=(0, 1))
    return soln.x

def minimax_LP_solver_v3(A: np.ndarray, b: np.ndarray, binary = False):
    """
    Solve Least Square problem below
    min_x max_i |A_i·x - b_i|
    where
    A_i : ith row vector of A (1-d array size m)
    b_i : ith element of b (scalar)
    x : [..., x_i, ...] (1-d array size m)
    0 ≤ x_i ≤ 1.
    Usually, n is number of class and m is number of training set.
    If binary is True, x_i is 0 or 1. 

    Args:
        A: given 2-d array size n ⨯ m
        b: given 1-d array size n
        binary: indicates the type of problem

    Return:
        x: np.array of solution
    """
    print(f"### Scipy minimax LP solver ###")
    if A.shape[0] != b.shape[0]:
        raise NotImplementedError
    
    n, m = A.shape
    c = np.zeros(m+1)
    c[-1] = 1

    A_ub = np.concatenate([np.concatenate([-A, A], axis=0), np.ones([2*n, 1])], axis=1)
    b_ub = np.concatenate([-b, b], axis=0)
    bounds = [(0, 1)]*m
    bounds.append((0, None))

    soln = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    if soln.success is not True:
        print(f"{soln.message=}")
        return minimax_LP_solver_v2(A, b, binary=binary)
    return soln.x[:-1]

def minsum_LP_solver_v3(A: np.ndarray, b: np.ndarray, binary = False):
    """
    Solve Least Square problem below
    min_x sum_i |A_i·x - b_i|
    where
    A_i : ith row vector of A (1-d array size m)
    b_i : ith element of b (scalar)
    x : [..., x_i, ...] (1-d array size m)
    0 ≤ x_i ≤ 1.
    Usually, n is number of class and m is number of training set.
    If binary is True, x_i is 0 or 1. 

    Args:
        A: given 2-d array size n ⨯ m
        b: given 1-d array size n
        binary: indicates the type of problem

    Return:
        x: np.array of solution
    """
    print(f"### Scipy minsum LP solver ###")
    """
    length of solution: m+2*n
    first m: x_i
    second n: y- (where a_i*x_i - b_i = y_i)
    third n: y+
    """
    if A.shape[0] != b.shape[0]:
        raise NotImplementedError
    
    n, m = A.shape

    bounds = [(0, 1)]*m
    bounds += [(0, None)]*(2*n)

    A_eq = np.concatenate([A, -np.eye(n), np.eye(n)], axis=1)
    c = np.concatenate([np.zeros(m), np.ones(2*n)])

    soln = linprog(c, A_eq=A_eq, b_eq=b, bounds=bounds)
    if soln.success is not True:
        print(f"{soln.message=}")
        return minsum_LP_solver_v2(A, b, binary=binary)
    return soln.x[:m]