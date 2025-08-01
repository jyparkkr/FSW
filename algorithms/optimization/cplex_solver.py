import numpy as np
from docplex.mp.model import Model
from docplex.mp.context import Context

def default_cplex_setting():
    # Create a context and set the 'threads' parameter
    context = Context.make_default_context()
    context.cplex_parameters.threads = 4  # Set the number of threads to 5


def LS_solver_v2(A: np.array, b: np.array, binary = False):
    default_cplex_setting()
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
    print(f"### Cplex LS solver ###")
    print(f"{A.shape=}")
    print(f"{b.shape=}")
    if A.shape[0] != b.shape[0]:
        print(f"{A.shape=}")
        print(f"{b.shape=}")
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

def absolute_minimax_LP_solver_v2(A: np.ndarray, b: np.ndarray, binary = False):
    default_cplex_setting()
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
    print(f"### Cplex absolute_minimax LP solver ###")
    if A.shape[0] != b.shape[0]:
        raise NotImplementedError
    
    n, m = A.shape
    model = Model('absolute_minimax_LP')
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
    soln = np.array([x[i].solution_value for i in range(m)])
    return soln

def absolute_minsum_LP_solver_v2(A: np.ndarray, b: np.ndarray, binary = False):
    default_cplex_setting()
    """
    Solve absolute Linear Programming below
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
        x: np.array of solution (length m)
    """

    print(f"### Cplex absolute_minsum LP solver ###")
    if A.shape[0] != b.shape[0]:
        raise NotImplementedError
    
    n, m = A.shape
    model = Model('absolute_minsum_LP')
    x_dict = (i for i in range(m))

    if binary:
        x = model.binary_var_dict(x_dict, name = 'x', lb = 0, ub = 1)
    else:
        x = model.continuous_var_dict(x_dict, name = 'x', lb = 0, ub = 1)
    
    y_dict = (i for i in range(2*n))
    # first n: y- (where a_i*x_i - b_i = y_i), second n: y+
    y = model.continuous_var_dict(y_dict, name = 'y', lb = 0)

    c1 = model.add_constraints([-y[i] + y[i+n] + sum(A[i,j]*x[j] for j in range(m)) == b[i] for i in range(n)], names = "eq_")

    obj = sum([y[i]+y[i+n] for i in range(n)])
    model.set_objective("min", obj)
    # model.print_information()
    # print(model.export_as_lp_string())
    model.solve(log_output=False) 
    soln = np.array([x[i].solution_value for i in range(m)])
    return soln

def absolute_and_nonabsolute_minsum_LP_solver_v1(
        A: np.ndarray, b: np.ndarray, C: np.ndarray, d: np.ndarray, binary = False):
    default_cplex_setting()
    """
    Solve absolute Linear Programming below
    min_x sum_i |A_i·x - b_i| + C_i·x - d_i
    where
    A_i : ith row vector of A (1-d array size m)
    b_i : ith element of b (scalar)
    Same for C and d
    x : [..., x_i, ...] (1-d array size m)
    0 ≤ x_i ≤ 1.
    Usually, n is number of class and m is number of training set.
    If binary is True, x_i is 0 or 1. 

    Args:
        A: given 2-d array size n ⨯ m
        b: given 1-d array size n
        C: given 2-d array size n ⨯ m
        d: given 1-d array size n
        binary: indicates the type of problem

    Return:
        x: np.array of solution (length m)
    """

    print(f"### Cplex absolute_and_nonabsolute_minsum LP solver ###")
    if A.shape[0] != b.shape[0]:
        raise NotImplementedError
    
    n, m = A.shape
    model = Model('absolute_and_nonabsolute_minsum__LP')
    x_dict = (i for i in range(m))

    if binary:
        x = model.binary_var_dict(x_dict, name = 'x', lb = 0, ub = 1)
    else:
        x = model.continuous_var_dict(x_dict, name = 'x', lb = 0, ub = 1)
    
    y_dict = (i for i in range(2*n))
    # first n: y- (where a_i*x_i - b_i = y_i), second n: y+
    y = model.continuous_var_dict(y_dict, name = 'y', lb = 0)
    c1 = model.add_constraints([-y[i] + y[i+n] + sum(A[i,j]*x[j] for j in range(m)) == b[i] for i in range(n)], names = "eq_")

    yy_dict = (i for i in range(n))
    yy = model.continuous_var_dict(yy_dict, name = 'yy', lb = 0)
    c2 = model.add_constraints([+yy[i] + sum(C[i,j]*x[j] for j in range(m)) == d[i] for i in range(n)], names = "eq2_")

    obj = sum([y[i]+y[i+n]+yy[i] for i in range(n)])
    model.set_objective("min", obj)
    # model.print_information()
    # print(model.export_as_lp_string())
    model.solve(log_output=False) 
    soln = np.array([x[i].solution_value for i in range(m)])
    return soln
