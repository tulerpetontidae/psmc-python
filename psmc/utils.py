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


def process_ts(ts, individual=None, start=None, end=None, window_size=100, progress=False):
    """
    Turn the variation data from a specific ``individual`` in a tree sequence into a
    numpy array indicating the presence or absence of heterozygotes in 100bp
    windows from ``start`` to ``end``. If ``individual`` is None, simply pick
    the individual associated with the first sample node
    """
    def is_connected(tree, node1, node2):
        """
        Check if neither node is isolated in the tree.
        """
        return not(tree.is_isolated(node1) or tree.is_isolated(node2))

    if not ts.discrete_genome:
        raise ValueError("Tree sequence must use discrete genome coordinates")
    if individual is None:
        individual = ts.node(ts.samples()[0]).individual
        if individual < 0:
            raise ValueError("No individual associate with the first sample node")
    try:
        nodes = ts.individual(individual).nodes
    except IndexError:
        raise ValueError(f"Individual {individual} not found in tree sequence")
    # Quickest to simplify to 2 genomes (gets rid of nonvariable sites etc)
    ts = ts.simplify(samples = nodes)
    if ts.num_samples != 2:
        raise ValueError(f"Individual {individual} did not have 2 genomes")
    if start is None:
        start = 0
    if end is None:
        end = int(ts.sequence_length)
    if (end-start) % window_size != 0:
        print(
            f"Warning: the genome size is not a multiple of {window_size}. "
            "The last window will be skipped."
        )

    result = np.empty((1, int((end-start) // window_size)), dtype=np.int8)
    # Processing is complicated because we want to look at windows even if they contains
    # non-variable sites. We check for missing data by looking the tree at each site.
    tree_iter = ts.trees()
    tree = next(tree_iter)
    variant = next(ts.variants(copy=False))  # get a Variant obj
    assert variant.site.id == 0

    # place the tree iterator and the variant iterator at the start
    use_trees = True
    while tree.interval.right < start and use_trees:
        if tree.index < ts.num_trees - 1:
            tree = next(tree_iter)
        else:
            use_trees = False

    use_variants = True
    while variant.site.position < start and use_variants:
        # could probably jump to the right start point here
        if variant.site.id < ts.num_sites - 1:
            variant.decode(variant.site.id + 1)
        else:
            use_variants = False

    # Now iterate through the windows
    seq = np.zeros(window_size, dtype=np.int8)
    wins = np.arange(start, end, window_size)
    for i, (left, right) in tqdm(
        enumerate(zip(wins[:-1], wins[1:])),
        total=len(wins) - 1,
        desc=f"Calc {window_size}bp windows",
        disable=not progress,
    ):
        # 0=missing, 1=homozygous, 2=heterozygous
        if not use_trees:
            seq[:] = 0
        else:
            while (tree.interval.right < right):
                tree_left = int(tree.interval.left)
                tree_right = int(tree.interval.right)
                if tree_left < left:
                    seq[0: tree_right - left] = is_connected(tree, 0, 1)
                else:
                    seq[tree_left - left: tree_right - left] = is_connected(tree, 0, 1)
                if tree.index == ts.num_trees - 1:
                    use_trees = False
                    seq[tree_right - left: window_size] = 0
                else:
                    tree = next(tree_iter)
            if use_trees:
                l_pos = max(int(tree.interval.left) - left, 0)
                seq[l_pos:window_size] = is_connected(tree, 0, 1)
        if np.count_nonzero(seq == -1) != 0:
            print(tree.index, seq)
            raise ValueError()
        while use_variants and variant.site.position < right:
            pos = int(variant.site.position) - left
            if (variant.has_missing_data):
                seq[pos] = 0
            elif variant.genotypes[0] == variant.genotypes[1]:
                seq[pos] = 1
            else:
                # heterozygous
                seq[pos] = 2
            if variant.site.id == ts.num_sites - 1:
                use_variants = False
            else:
                variant.decode(variant.site.id + 1)
        if np.count_nonzero(seq == 0) >= int(window_size * 0.9):
            result[0, i] = 2  # "N"
        elif np.count_nonzero(seq == 2) > 0:
            result[0, i] = 1  # "K" = at least one heterozygote
        else:
            result[0, i] = 0  # "T" = all homozygous
    return result


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

