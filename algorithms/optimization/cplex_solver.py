import torch
import numpy as np
from docplex.mp.model import Model

def LS_solver(A: np.array, b: np.array, binary = False):
    """
    Solve Least Square problem below
    min_x (Ax - b)**2 
    where
    x: [..., x_i, ...] (1-d array size m)
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
    if A.shape[0] != b.shape[0]:
        raise NotImplementedError
    
    n, m = A.shape
    model = Model('Sample')
    x_dict = (i for i in range(m))
    if binary:
        x = model.binary_var_dict(x_dict, name = 'x', lb = 0, ub = 1)
    else:
        x = model.continuous_var_dict(x_dict, name = 'x', lb = 0, ub = 1)

    obj = sum((sum(A[i,j]*x[j] for j in range(m)) - b[i])**2 for i in range(n))
    model.set_objective("min", obj)

    model.solve(log_output=False)
    soln = np.array([x[i].solution_value for i in range(m)])
    return soln


def minimax_LP_solver(*args, **kwargs):
    return minimax_LP_solver_v2(*args, **kwargs)


def minimax_LP_solver_v2(A: np.ndarray, b: np.ndarray, binary = False):
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
    if A.shape[0] != b.shape[0]:
        raise NotImplementedError
    
    n, m = A.shape
    model = Model('minimax_LP')
    x_dict = (i for i in range(m))

    if binary:
        x = model.binary_var_dict(x_dict, name = 'x', lb = 0, ub = 1)
    else:
        x = model.continuous_var_dict(x_dict, name = 'x', lb = 0, ub = 1)
    y = model.continuous_var(name = 'y', lb = 0)

    c1 = model.add_constraints([-y + sum(A[i,j]*x[j] for j in range(m)) <=   b[i] for i in range(n)], names = "ub_")
    c2 = model.add_constraints([-y - sum(A[i,j]*x[j] for j in range(m)) <= - b[i] for i in range(n)], names = "lb_")

    model.set_objective("min", y)
    # model.print_information()
    # print(model.export_as_lp_string())
    model.solve(log_output=False) 
    soln = [x[i].solution_value for i in range(m)]
    # print("Objective value : " , model.objective_value )    # 목적함수값 출력
    return soln