from scipy.special import logsumexp
import numpy as np
import re
import gzip
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

def process_psmcfa(psmcfa, batch_size=300000):
    # Convert a psmcfa file to a numpy array. PSMCFA files are fasta-like files of the form
    # > chr1
    # NNNKTTTKTNKTT
    # where each letter represents a non-overlapping bin (of e.g. 100bp) indicating if there is at least
    # one heterozygote in each bin (K), all are homozygous (T) or the bin should be treated as missing
    # (e.g. >=90 filtered bases)
    if psmcfa.endswith('.gz'):
        with gzip.open(psmcfa, 'rt') as file:
            psmcfasta = file.read().replace('\n', '')
    else:
        with open(psmcfa, 'r') as file:
            psmcfasta = file.read().replace('\n', '')
    generated_seq = [re.sub(r'^\d+', '', x) for x in psmcfasta.split('>')[1:]]
    
    allowed_chars = set(['T', 'K', 'N'])

    data = []
    for string in generated_seq:
        new_string = ''.join(c for c in string if c in allowed_chars)
        data.append(new_string)

    if batch_size is None:
        for batch_id in range(len(data)):
            data[batch_id] = list(data[batch_id].replace('T', '0').replace('K', '1').replace('N', '2'))
        data = np.array(data).astype(int)

        return data
    else:
        data = ''.join(data)
        residual = len(data) % batch_size
        data = data.replace('T', '0').replace('K', '1').replace('N', '2')
        data = data + '2' * (batch_size - residual)
        data = np.array(list(data)).astype(int)
        return data.reshape(-1, batch_size)

