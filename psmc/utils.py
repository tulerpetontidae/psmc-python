from scipy.special import logsumexp
import numpy as np
import re
from scipy.optimize import minimize
from tqdm.notebook import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(x):
    return  np.log(x/(1-x))

def log_domain_matmul(log_A, log_B):
    """
    log_A : ... x n
    log_B : n x ...
    output : ... x ... matrix

    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
    """

    log_A_ndims = len(log_A.shape)
    log_B_ndims = len(log_B.shape)
        
    log_A_expanded = log_A.reshape(list(log_A.shape) + [1]*(log_B_ndims - 1))
    log_B_expanded = log_B.reshape([1]*(log_A_ndims - 1) + list(log_B.shape))

    elementwise_sum = log_A_expanded + log_B_expanded
    out = logsumexp(elementwise_sum, axis=log_A_ndims-1)
    return out

# NOT GENERALIZED
def maxmul(log_A, log_B):
    """
    log_A : m x n
    log_B : n x p
    output : m x p matrix; maxval, argmaxval

    Similar to the log domain matrix multiplication,
    this computes out_{i,j} = max_k log_A_{i,k} + log_B_{k,j}
    """
    m = log_A.shape[0]
    n = log_A.shape[1]
    p = log_B.shape[1]

    log_A_expanded = np.stack([log_A] * p, axis=2)
    log_B_expanded = np.stack([log_B] * m, axis=0)

    elementwise_sum = log_A_expanded + log_B_expanded
    out1, out2 = np.max(elementwise_sum, axis=1), np.argmax(elementwise_sum, axis=1)
    
    return out1,out2

def read_sim_history(s):
    pattern = r'-eN\s+([\d.]+)\s+([\d.]+)'
    matches = re.findall(pattern, s)
    tuples = [[float(match[0]), float(match[1])] for match in matches]
    tuples = [[0, tuples[0][1]]] + tuples
    return np.array(tuples)

def tqdm_minimize(fun, x0, args=(), method=None, jac=None, hessp=None,
                  hess=None, constraints=(), tol=None, callback=None,
                  options=None, bounds=None):
    """
    A modified version of the `minimize` function from scipy.optimize that prints a tqdm-like
    output on each iteration.

    Args:
        fun: The objective function to be minimized.
        x0: Initial guess for the optimization parameters.
        args: Extra arguments to be passed to the objective function.
        method: Name of the optimization method to use.
        jac: Function to calculate the Jacobian matrix (gradient) of the objective function.
        hessp: Function to calculate the Hessian matrix of the objective function multiplied by a vector.
        hess: Function to calculate the Hessian matrix of the objective function.
        constraints: A list of constraint dictionaries or instances of LinearConstraint or NonlinearConstraint.
        tol: Tolerance for termination.
        callback: A function called after each iteration of the optimization.
        options: A dictionary of solver options.
        bounds: A sequence of (min, max) pairs for each element in x.

    Returns:
        A `OptimizeResult` object representing the optimization result.

    """
    if options is None:
        options = {}

    n_iter = options.get('maxiter', 200)
    t = tqdm(total=n_iter)

    def callback_wrapper(x):
        t.update(1)

    return minimize(fun, x0, args=args, method=method, jac=jac, hessp=hessp, hess=hess,
                    constraints=constraints, tol=tol, callback=callback_wrapper,
                    options=options, bounds=bounds)
