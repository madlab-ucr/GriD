# -*- coding: utf-8 -*-         # needed for "degree" symbols in comments later
"""
Various helper functions that are used by the main methods of SSL-H


Nomenclature:
    X, Xc, X0: [n x k] np.array of normalized (centered, explicit) beliefs
    Xd: node id -> class id dictionary
    H, Hc: [k x k] np.array of normalized (centered) compatibility matrix
    W: sparse edge matrix

(C) Wolfgang Gatterbauer, 2016
"""


from __future__ import print_function
from __future__ import division
from scipy.sparse import csr_matrix, kron, diags, issparse, identity
from scipy.optimize import newton
# from pyamg.util.linalg import approximate_spectral_radius   #pyamg unreliable find, load, and execute. Function now replaced with 'scipy.sparse.linalg.eigs'
from scipy.sparse.linalg import eigs, eigsh                 # alternative spectral radius method
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import normalize, scale
from itertools import product
from math import acos, pi
import collections
import warnings
import numpy as np
PREC = 1e-4                 # defined precision for checking sum = 1+-PREC
SPECTRAL_TOLERANCE = 1e-8   # if std of a centered matrix is too small, then those are basically rounding errors




def check_normalized_beliefs(X):
    """Verifies that rows in a [n x k] np.array of beliefs are valid probability distributions
    Allows n=1, thus one dimensional vector [ , ] in addition to 2-dimensional [[ , ]]
    """
    assert type(X).__module__ == "numpy"    # make sure it is not a matrix
    correct = X.all() >= 0
    if len(X.shape) == 1:                   # special case in case n=1 and no nesting: [[...]] -> [...]
        return correct and np.abs(np.sum(X)-1) <= PREC
    else:
        return correct and (np.abs(np.sum(X,1)-1) <= PREC).all()


def check_centered_beliefs(Xc):
    """Verifies  that rows in a [n x k] np.array of centered beliefs sum up to 0
    Allows [ , ] in addition to [[ , ]] for n=1
    """
    assert type(Xc).__module__ == "numpy"   # make sure it is not a matrix
    if len(Xc.shape) == 1:                  # special case in case n=1 and no nesting: [[...]] -> [...]
        return np.abs(np.sum(Xc)) <= PREC
    else:
        return (np.abs(np.sum(Xc,1)) <= PREC).all()


def check_dictionary_beliefs(Xd, n=None, k=None, zeroindexing=True, fullBeliefs=False):
    """Verify that the explicit belief dictionary Xd (node id -> class id) has certain properties
    fullBeliefs = True: every node has at least one belief
    zeroindexing  = True if first node is indexed by 0 [instead of by 1] (only for fullBeliefs)
    """
    keys = Xd.keys()
    values = set(Xd.values())
    v0 = min(keys)
    v1 = max(keys)
    n2 = len(Xd.keys())                     # number of node assignments
    j0 = min(values)
    j1 = max(values)
    k2 = len(values)                        # number of different classes
    if zeroindexing:
        correct = v0 >=0
        delta = 0
    else:
        correct = v0 >=1
        delta = -1
    if fullBeliefs:
        correct = correct and n2 == len(set(Xd.keys()))
        correct = correct and v0 == - delta                # minimum node id = 0 (or 1)
        correct = correct and j0 == - delta
        correct = correct and v1 == n2 - 1 - delta         # maximum node id = length - 1 (or 0)
        correct = correct and j1 == k2 - 1 - delta         # maximum class id = size - 1 (or 0)
        if n is not None:
            correct = correct and n == n2
        if k is not None:
            correct = correct and k == k2
    if n is not None:
        correct = correct and n >= v1 + 1 + delta
    if k is not None:
        correct = correct and k >= j1 + 1 + delta
    return correct


def check_explicit_beliefs(X):
    """Verifies that a given [n x k] np.array of beliefs assigns each node to maximal one class with 100%.
    Thus, each row has maximum one 1-entry per row, and the rest are all 0s.
    Allows [ , ] in addition to [[ , ]] for n=1
    """
    assert type(X).__module__ == "numpy"        # make sure it is not a matrix
    correct = set(np.hstack(X)) == set([0, 1]), "Input X can contain only 0 or 1 (as int or float)"
    if len(X.shape) == 1:                       # special case in case n=1 and no nesting: [[...]] -> [...]
        return correct and np.abs(np.sum(X)) <= 1
    else:
        return correct and (np.abs(np.sum(X,1)) <= 1).all()


def to_centering_beliefs(X, ignoreZeroRows=False):
    """Centers an [n x k] np.array of valid normalized beliefs around 1/k.
    Allows [ , ] in addition to [[ , ]] for n=1.
    If ignoreZeroRows: then ignore keep with only zeros as zeros.
    """
    # assert check_normalized_beliefs(X)    # statement would prevents using explicit belief vector
    assert type(X).__module__ == "numpy"    # make sure it is not a matrix
    if len(X.shape) == 1:                   # special case in case n=1 and no nesting: [[...]] -> [...]
        k = len(X)
    else:
        (_, k) = X.shape
    if ignoreZeroRows:
        Xl = to_explicit_bool_vector(X)
        Xl2 = np.array([1./k * Xl]).transpose()
        return X - Xl2
    else:
        return X - 1./k


def from_centering_beliefs(Xc):
    """Uncenteres a centered [n x k] np.array
    Allows [ , ] in addition to [[ , ]] for n=1
    Also used for compatibility matrix H
    """
    assert check_centered_beliefs(Xc)
    if len(Xc.shape) == 1:           # special case in case n=1 and no nesting: [[...]] -> [...]
        k = len(Xc)
    else:
        (_, k) = Xc.shape
    return Xc + 1./k


def to_dictionary_beliefs(X):
    """Transforms explicit belief assignment as [n x k] matrix to dictionary (node id -> belief id)
    Assumes zeroindexing
    requires nested [[,]]
    http://stackoverflow.com/questions/29301899/iterate-over-nested-arrays-storing-indexes
    """
    check_explicit_beliefs(X)       # verifies that appropriate belief matrix
    assert len(X.shape) == 2        # requires nested [[,]]
    Xd = {i[0] : i[1] for i in product(*[range(dim) for dim in X.shape]) if X[i] > 0}
    return Xd


def from_dictionary_beliefs(Xd, n=None, k=None, zeroindexing=True):
    """Takes an explicit belief assignment from dictionary format (node id -> belief id) and transforms it into a [n x k] belief matrix.
    If n=None, assumes node (n-1) [or n for zeroindexing=False] appears.
    If k=None, assumes class (k-1) [or k for zeroindexing=False] appears.
    """
    check_dictionary_beliefs(Xd, n=n, k=k, zeroindexing=zeroindexing)

    keys = Xd.keys()
    values = Xd.values()
    if zeroindexing:
        delta = 0
    else:
        delta = -1

    if n is None:
        n = max(keys) + 1 + delta
    if k is None:
        k = max(values) + 1 + delta

    X = np.zeros((n, k), dtype=np.int)
    for k,v in Xd.items():
        X[k+delta,v+delta] = 1
    # for i in range(n):
    #     X[i,Xd[i]] = 1
    return X


def to_explicit_bool_vector(X):
    """Returns a n-dimensional Boolean np.array that indicates which nodes have some beliefs <> 0.
    Every row with sum(abs(entries)) > 0 is assumed to be explicit.
    Input: [n x k] np array
    Returns: Boolean n-dimensional np array
    """
    Xb = np.array(np.sum(np.abs(X), 1) > PREC).flatten()     # matrix -> ndarray, then flatten
    return Xb


def to_explicit_list(X):
    """Returns a list of indices of nodes with explicit beliefs.
    Assumes a [n x k] np.array of normalized or explicit centered beliefs X with n > 1.
    every row with sum(abs(entries)) > 0 is assumed to be explicit.
    Thus only checks for having some entries different from 0 and therefore accepts both centered and non-centered beliefs
    """
    s = np.array(np.sum(np.abs(X), 1)).flatten()             # simple flat array (not matrix) of absolute row sums
    Xl = [i for i, j in enumerate(s) if j > PREC]
    return Xl


def max_binary_matrix(X, threshold=0):
    """Returns a [n x k] np.array binary integer matrix for top class for each node
    Assumes a [n x k] np.array matrix X; also allows a threshold value for declaring winner
    Makes sure that matrix has 2 dimensions
    """
    assert type(X).__module__ == "numpy"            # make sure it is not a matrix
    X = np.atleast_2d(X)                            # require 2d arrays
    X2 = X.max(1, keepdims=True) - threshold        # creates vertical [[],] vector of max per row
    X3 = 1*(X >= X2)                                # creates integer
    return X3


def row_normalize_matrix(M, norm='l1'):
    """Normalizes or standardizes each row of a two-dimensional array (not necessarily numpy array)
    Allows L2 or L1 norms ('l1' or 'l2'), or 'zscores' ("scaling").
    Allows rows with zero vectors.
    Serves as wrapper around sklearn.preprocessing.normalize and sklearn.preprocessing.scale functions.
    Fixes following issues:
        (1) allows int in addition to float
        (2) allows single rows as input
    """
    assert not issparse(M), "Matrix cannot to be sparse"
    M = np.atleast_2d(M)    # both preprocessing.normalize and zscore require 2d arrays
    M = M.astype(float, copy=False)

    if norm in ['l2', 'l1']:
        return normalize(M, axis=1, norm=norm)
    elif norm == 'zscores':
        return scale(M, axis=1, with_mean=True, with_std=True, copy=True)


def calculate_potential_from_row_normalized(H, alpha, d_vec=None):
    """Calculates the normalized potential 'P' (sum all entries = 1).
    Takes as input a (directed) row-normalized matrix 'H' and an (outgoing) label distribution 'alpha'.
    Optionally takes an average degree vector per class 'd_vec'. If not provided (= None), assumes the same across classes
    """
    assert type(H).__module__ == "numpy"       # make sure it is not a matrix
    alphaT = np.array([alpha]).transpose()
    if d_vec is None or not isinstance(d_vec, (collections.Sequence, np.ndarray)):
        # collections.Sequence is supertype of list, np.array needs to be treated separately
        return H * alphaT
    else:
        P = H * alphaT * np.array([d_vec]).transpose()   # multiply with relative average degree per class
        return 1. * P / sum(P.flatten())        # Potential: normalized sum = 1


def degree_matrix(W, indegree=True, undirected=False, squared=True):
    """Calculates diagonal in- or out-degree matrices.
    W must be sparse. W can be weighted matrix.
    Considers two very different cases:

    1. squared=False:
    Application: e.g., we need the degree-matrix for calculating the row-normalized matrix
    Either standard in- or out-degree matrix.
    (if undirected, then 'indegree' is ignored and indegree is calculated, since in- and out-degree are the same)

    2. squared=True:
    Application: the LinBP echo cancellation term: weights are squared before summed (as messages go in both directions)
    2a. If undirected=False (matrix is directed)
    - D_in: indegree=True (sum of squared col entries)
    - D_out: indegree=False (sum of squared row entries)
    2b. If undirected=True
    - Then the weight of an edge can be different in either direction (e.g., for 'W_row').
    - Then degree is the sum of the weights multiplied in both directions which is the same in both directions.
    - Then 'indegree' is ignored (intermediate matrix W2 is symmetric).
    - This option returns empty matrix in case of a directed graph (with no back edges)
    """
    assert issparse(W), "Matrix needs to be sparse"
    n, _ = W.shape
    W2 = W
    if squared:
        if undirected:
            W2 = W.multiply(W.transpose())  # entrywise multiplication
        else:
            W2 = csr_matrix(W, copy=True)
            W2.data[:] = np.power(W2.data, 2)

    if indegree:
        degree = W2.transpose().dot(np.ones([n]))       # sum up all the weighted edges (optionally after squaring)
    else:
        degree = W2.dot(np.ones([n]))
    return diags(degree, 0)


def create_parameterized_H(k, h, symmetric=True):
    """Constructs a symmetric doubly stochastic matrix H with 'k' dimensions that has some heterophily.
    The matrix has two entries: high and low. High is 'h' times higher than low.
    If symmetric==False, then constructs a slight asymmetric variation (for directed edges)
    """
    low = 1 / (k - 1 + h)
    high = low * h
    H = np.full((k, k), low)
    if symmetric:
        for j in range(0, k//2):
            H[2*j,2*j+1] = high
            H[2*j+1,2*j] = high
        if (k % 2) == 1:
            H[k-1, k-1] = high
    else:
        for j in range(0, k-1):
            H[j,j+1] = high
        H[k-1, 0] = high
    return H


def create_parameterized_alpha(k, f):
    """Constructs a 'k'-dimensional stochastic vector 'alpha' (sum is equal to 1)
    The highest entry is 'f' times higher than low.
    The first entry is low, the last one high. Everything between is linearly interpolated.
    """
    low = 2 / (k*(1+f))
    delta = low * (f-1) / (k-1)
    alpha = [low + i*delta for i in range(0,k)]
    return np.array(alpha)


def W_row(W):
    """Constructs a row-normalized matrix 'W_row' from 'W'.
    Requires 'W' to be sparse. Works with weighted matrix
    """
    assert issparse(W), "Matrix needs to be sparse"
    D_inv = degree_matrix(W, squared=False, indegree=False)        # row-normalized: thus D_inv contains sum of outdegrees on diagonal

    D_inv.data[:] = np.power(D_inv.data, -1)    # inverse
    return D_inv.dot(W)


def W_red(W):
    """Constructs a symmetric semi-normalized matrix 'W_red' from 'W'.
    Requires 'W' to be sparse and symmetric.
    TODO: think about weighted asymmetric graph.
    """
    assert issparse(W), "Matrix needs to be sparse"
    D_inv = degree_matrix(W, squared=False, indegree=False)         # Assumes undireced graph, thus in-degree = out-degree.
    D_inv.data[:] = np.power(D_inv.data, -0.5)  # D^(-1/2)
    return D_inv.dot(W).dot(D_inv)


def W_clamped(W, indices):
    """Given a sparse matrix 'W' and an array_like list 'indices' (that indicates all explicit nodes)
    Removes all rows with index contained in indices (thus all incoming edges to explicit nodes)
    """
    assert issparse(W), "Matrix needs to be sparse"
    (n, _) = W.shape
    row, col = W.nonzero()
    weight = W.data
    mask = np.in1d(row, indices, invert=True)   # find the indices in 'row' that are not contained in 'indices'
    return csr_matrix((weight[mask], (row[mask], col[mask])), shape=(n, n))


def W_star(W, alpha=0, beta=0, gamma=0, indices=[]):
    """Given a sparse matrix 'W' and an optional array_like list 'indices' (that indicates all explicit nodes)
    Calculates the resulting propagation matrix according to paper formula
    Generalizes all propagation matrices:
    W:      alpha=0, beta=0
    W_row:  alpha=1, beta=0
    W_red:  alpha=0.5, beta=0.5
    clamped: gamma=1, and indices specified
    """
    assert issparse(W), "Matrix needs to be sparse"
    row, col = W.nonzero()  # Warning in case matrix is not symmetric (even after ignoring weights, e.g., for W_row)
    if not (set(zip(row, col)) == set(zip(col, row))):  # Thus, warning appears e.g. for W_clamped
        warnings.warn("\nEdges in W are not undirectional", UserWarning)

    D_inv = degree_matrix(W, squared=False, indegree=False)        # row-normalized: thus D_inv contains sum of outdegrees on diagonal
    n, _ = W.shape

    if alpha == 0:
        D1 = identity(n)
    else:
        D1 = csr_matrix(D_inv, copy=True)
        D1.data[:] = np.power(D1.data, - alpha)

    if beta == 0:
        D2 = identity(n)
    else:
        D2 = D_inv
        D2.data[:] = np.power(D2.data, - beta)

    if gamma == 0:
        C = identity(n)
    else:
        weight = np.ones(len(indices))
        C_minus = csr_matrix((weight, (indices, indices)), shape=(n, n))
        C = identity(n) - C_minus.dot(gamma)

    return C.dot(D1).dot(W).dot(D2)



def row_recentered_residual(P, paperVariant=True):
    """Row-recenters a given potential (normalized or not) in 2 variants.
    Assumes np array and that every entry is > 0
    Returns np array

    Variant 1 (paperVariant=True): as currently defined in the figure: P -> ^P -> ^P' -> P'
    Variant 2: alternative version if one knows the total distribution of vertices labels in graph
    """


    assert type(P).__module__ == "numpy"       # make sure it is not a matrix
    if paperVariant:
        l, k = P.shape
        P_tot = sum(P.flatten())        # sum of all entries.
        P = 1. * P / P_tot * k * l      # now centered around 1 (sum = k*l)

        Pc = P - 1                      # now residual
        Pc_rowsum = Pc.sum(1, keepdims=True)    # creates vertical [[],] vector of rowsums
        Pc_row = 1./k *(Pc - 1.*Pc_rowsum/k)    # row-recentered residual according to paper
        return Pc_row
    else:
        P_row = row_normalize_matrix(P, norm='l1')
        Pc_row = to_centering_beliefs(P_row)
        return Pc_row


# === ACCURACY ========================================================================


def matrix_difference(X, Y, ignore_rows=list([]), similarity='accuracy', vector=False):
    """Calculate difference (or similarity) between two [n x k] matrices X (ground truth) and Y (predicted).
    Difference is calculated row-wise (for each node separately). Except for option similarity='fro'
    Optional argument specifies which rows should be ignored.
    (e.g., because they were labeled with explicit beliefs, and we calulate accuracy for unlabeled nodes).
    Allows to return a n-dimensional vector with row-wise differences or a single average over all rows.
    Allows 'accuracy', 'precision', 'recall', 'l2' differences, 'cosine', 'cosine_ratio' similarities.
    For 'accuracy', 'precision', 'recall': compares for each row in GT and Method, the classes with top beliefs.

    Uses: max_binary_matrix()

    Parameters
    ----------
    X : [n x k] np array
        true belief matrix (GT)
    Y : [n x k] np array
        predicted belief matrix
    ignore_rows : int list, optional (Default=empty)
        list of rows to ignore [list of explicit beliefs that are not evaluated]
    similarity : what type of similarity function used, optional (Default='accuracy')
        similarity='accuracy' : "In multilabel classification, this function computes subset accuracy:
            the set of labels predicted for a sample must *exactly* match the
            corresponding set of labels in X." Copied from [sklearn.metrics.accuracy_score]
        similarity='precision' :
        similarity='recall' :
        similarity='cosine' : Compute cosine similarity between rows in [n x k] arrays X and Y.
            Returns a 1-d array with n entries, one for each row.
            Cosine similarity, or the cosine kernel, computes similarity as the
            normalized dot product of X and Y:
                K(X, Y) = <X, Y> / (||X||*||Y||)
            [Inspired by sklearn.metrics.pairwise.cosine_similarity, but returns simple vector instead of matrix]
        similarity='cosine_ratio' : Use cosine similarity as before, but then also multiply with ratio of lengths.
            In other words, this is the ratio of the projection of the smaller vector onto the larger vector.
            Solves the problem if both vectors point in same direction but are of different lengths (not yet converged)
        similarity='l2' : Compute L2 difference between rows in [n x k] arrays X and Y.
        similarity='l2_matrix' : Computes L2 (Frobenius) *matrix* norm, thus ignores the vector attribute
    vector : bool, optional (Default=False)
        if True, then returns the vector of individal accuracies per row instead of the average over all

    Returns
    -------
    accuracy :  float, or
                numpy.ndarray(floats)
    """
    # verify input
    if similarity not in ('accuracy', 'precision', 'recall', 'cosine','cosine_ratio', 'l2', 'l2_matrix'):
        raise ValueError("'%s' is not a supported similarity function" % similarity)
    X = np.atleast_2d(X)    # needed for using the shape function below
    Y = np.atleast_2d(Y)
    n, k = X.shape
    n2, k2 = Y.shape
    assert(n == n2), "Matrices need to have the same dimensions"
    assert(k == k2), "Matrices need to have the same dimensions"

    # use list ignore_rows to ignore certain rows
    indTest = list(set(range(n)).difference(set(ignore_rows)))   # evaluate accuracy only on implicit beliefs
    X = np.asarray(X[indTest])
    Y = np.asarray(Y[indTest])

    if similarity in ['cosine', 'cosine_ratio']:
        Xn = row_normalize_matrix(X, norm='l2')
        Yn = row_normalize_matrix(Y, norm='l2')
        Z = np.array([np.dot(Xn[i, :], Yn[i, :]) for i in range(Xn.shape[0])])
            # new n = first entry of shape (after ignoring explicit belief rows)
        if similarity == 'cosine_ratio':
            Xl = np.linalg.norm(X, axis=1)  # calculate length of each row-vectors
            Yl = np.linalg.norm(Y, axis=1)  # calculate length of each row-vectors
            Zl = np.array([i/j if i<j else j/i if j<i else 1 for (i,j) in zip(Xl,Yl)])
            Z = Z*Zl
    elif similarity == 'l2':
        Z = np.linalg.norm(X-Y, axis=1)
    elif similarity == 'l2_matrix':                   # ignore the vector parameter and always return a single value
        Z = np.linalg.norm(X-Y, axis=None)
        return Z
    else:
        THRESHOLD = 1e-8  # threshold for binarization
        Xb = max_binary_matrix(X, threshold=THRESHOLD)  # GT: binary matrices indicating one or more top beliefs per row
        Yb = max_binary_matrix(Y, threshold=THRESHOLD)  # predicted:
        Zb = Xb * Yb  # overlap

        x = Xb.sum(axis=1)  # sum up the rows of binary matrices
        y = Yb.sum(axis=1)
        z = Zb.sum(axis=1)

        if similarity == 'precision':
            Z = 1. * z / y
        elif similarity == 'recall':
            Z = 1. * z / x
        elif similarity == 'accuracy':
            Z = 1. * ( (Xb != Yb).sum(axis=1) == 0 )
    if vector:
        return Z
    else:
        return np.average(Z)


def matrix_convergence_percentage(X0, X1, threshold=0.9962, ignore_rows=list([]), similarity='cosine_ratio'):
    """Takes two 2d numpy arrays and returns the percentage of rows that have similarity > threshold (or difference < threshold).
    Thus returns the percentage of beliefs that are close to converging.
    If threshold == None, then:
        returns the average similarity score across all rows (for 'cosine' and 'cosine_ratio')
        returns the frobenius (l2) matrix siilarity (for 'l2')
        ignored for 'accuracy'
    Optional argument specifies which rows should be ignored
        (e.g., because they were labeled with explicit beliefs, and we only calulate accuracy for unlabeled nodes).
    Two rows with only zeros are assumed to be perpendicular
        (thus nodes without any beliefs in early iterations are assumed not to have yet converged).
    An important assumption is thus that every connected component has at least one node with explicit beliefs).
    It is recommended to first standardize the input matrices before using this function.

    Uses: matrix_difference()

    Parameters
    ----------
    X0 : [n x k] np array
        first matrix
    Y1 : [n x k] np array
        second matrix
    ignore_rows : int list, optional (Default=empty)
        list of rows to ignore [list of explicit beliefs that are not evaluated]
    threshold : float (Default = 0.9962)
        threshold cosine similarity between rows in [n x k] arrays X and Y.
            0.939692621: 20°, 0.965925826: 15°, 0.984808: 10°, 0.996194: 5°, 0.999391: 2°, 0.999848: 1°
        threshold l2 norm between rows
    similarity : what type of similarity function used, optional (Default='cosine_ratio')
        similarity='accuracy' : "In multilabel classification, this function computes subset accuracy:
            the set of labels predicted for a sample must *exactly* match the
            corresponding set of labels in X." Copied from [sklearn.metrics.accuracy_score]
        similarity='cosine' : Compute cosine similarity between rows in [n x k] arrays X and Y.
            Returns a 1-d array with n entries, one for each row.
            Cosine similarity, or the cosine kernel, computes similarity as the
            normalized dot product of X and Y:
                K(X, Y) = <X, Y> / (||X||*||Y||)
            [Inspired by sklearn.metrics.pairwise.cosine_similarity, but returns simple vector instead of matrix]
        similarity='cosine_ratio' : Use cosine similarity as before, but then also multiply with ratio of lengths.
            In other words, this is the ratio of the projection of the smaller vector onto the larger vector.
            Solves the problem if both vectors point in same direction but are of different lengths (not yet converged)
        similarity='l2' : Compute L2 difference between rows in [n x k] arrays X and Y.
        similarity='l2_matrix' : Computes L2 (Frobenius) *matrix* norm. Thus ignores the threshold attribute and just calculates the matrix similarity
    comparison :
        comparison = 'threshold'

        comparison = 'score'

    Returns
    -------
    result_fractionce :  float
        Fraction of rows with cosine similarity (between two input matrices) > threshold
        Or Frobenius matrix norm for similarity='fro'
    """
    if similarity not in ('accuracy', 'cosine','cosine_ratio', 'l2'):
        raise ValueError("'%s' is not a supported similarity function in 'matrix_convergence_percentage'" % similarity)

    if threshold is None or similarity=='accuracy':
        return matrix_difference(X0, X1, similarity=similarity, vector=False, ignore_rows=ignore_rows)
    else:
        result_vector = matrix_difference(X0, X1, similarity=similarity, vector=True, ignore_rows=ignore_rows)
        if similarity=='l2':
            result_fraction = sum(1*(result_vector < threshold)) / len(result_vector)       # for cosine similarity, fraction that is more similar
        else:
            result_fraction = sum(1 * (result_vector > threshold)) / len(result_vector)     # for l2 norm, fraction that is less different
        return result_fraction



def angle_triangle(a, b, c):
    """Calculates the angle (in degrees) in a triangle with sides a, b, c where the angle is facing side c
    """
    COS = 1. * (a**2 + b**2 - c**2) / (2*a*b)
    gamma = acos(COS)
    return gamma*180/pi


# =======================================================================================================================================


def replace_fraction_of_rows(X0, f, avoidNeighbors=False, W=None, ind_prior=None):
    """Given [n x k] matrix. Replace a random fraction f of rows with 0-vector (i.e. replace exactly round(f*n) rows).
    Returns new [n x k] array, plus indices of remaining explicit beliefs (i.e. (1-f)*n of the remaining rows).
    previousindGT: Allows to specify another index list (with size bigger than (1-f)*n). Then only removes further edges from those.
        Used for experiments in which increasingly many labeled nodes are removed
    """
    # TODO: create variant that leaves no neighbors connected
    n, _ = X0.shape
    r = int(round(f*n))
    if not avoidNeighbors:
        if ind_prior is None:
            ind = np.random.choice(n, r, replace=False)         # index of replaced rows
            X = np.array(X0)
            X[ind, :] = 0
            indGT = list(set(range(n)).difference(set(ind)))    # index of unchanged rows
            indGT.sort()
            return X, indGT
        else:
            keep = int(round((1-f)*n))          # number of rows to keep
            n2 = len(ind_prior)
            assert keep <= n2                   # assert that the number is smaller than the remaining rows in the given ind_prior

            np.random.shuffle(ind_prior)
            indGT2 = ind_prior[:keep]
            indGT2.sort()                       # picks the remaining rows

            ind2 = list(set(range(n)).difference(set(indGT2)))  # calculates the inverse: rows to remove (including those that were already removed)
            X = np.array(X0)
            X[ind2, :] = 0
            return X, indGT2
    else:
        None




def introduce_errors(X1, ind, f):
    """Introduces a fraction 'f' of errors in an explicit belief vector.
    X1 : [n x k] belief matrix,
    ind : list of indices of the explicit beliefs (e.g., X1 can contain 90% zero columns, ind lists all the others)
    A fraction 'f' of the explicit rows indexed by 'ind' will be randomly permuted (and each permutation is checked to be different from the original one)
    """
    X2 = X1.copy()                      # important, otherwise X1 entries are overwritten
    m = int(round(f * len(ind)))
    ind = np.random.permutation(ind)    # permutation makes a copy (in contrast to shuffle)
    ind = ind[0:m]
    for i in ind:
        seq = np.random.permutation(X2[i])
        while np.array_equal(seq, X2[i]):       # makes sure that the permutation is not by chance the original one
            seq = np.random.permutation(X2[i])
        X2[i] = seq
    return X2



# === Spectral radius ==================================================================================================================


def approx_spectral_radius(M, pyamg=False, symmetric=False, tol=1e-04):
    """pyamg=False ... DEPRECATED
    Wrapper around existing methods to calculate spectral radius.
    1. Original method: function 'pyamg.util.linalg.approximate_spectral_radius'.
    Behaved strange at times, and packages needed time to import and returned errors.
    But kept as default since the other method from scipy sometimes gives wrong results!
    2. 'scipy.sparse.linalg.eigs' which seemed to work faster and apparently more reliably than the old method.
    However, it sometimes does not return the correct value!
    This happens when echo=True and the biggest value is negative. Then returns the next smaller positive.
    For example: returns 0.908 for [ 0.9089904+0.j -1.0001067+0.j], or 0.933 for [ 0.93376532+0.j -1.03019369+0.j]

    http://scicomp.stackexchange.com/questions/7369/what-is-the-fastest-way-to-compute-all-eigenvalues-of-a-very-big-and-sparse-adja
    http://www.netlib.org/utk/people/JackDongarra/etemplates/node138.html

    Both methods require matrix to have float entries (asfptype)
    Testing: scipy is faster up to at least graphs with 600k edges
    10k nodes, 100k edges: pyamp 0.4 sec, scipy: 0.04
    60k nodes, 600k edges: pyam 2 sec, scipy: 1 sec

    Allows both sparse matrices and numpy arrays: For both, transforms int into float structure.
    However, numpy astype makes a copy (optional attribute copy=False does not work for scipy.csr_matrix)

    'eigsh' is not consistently faster than 'eigs' for symmetric M
    """
    pyamg=False
    if pyamg:
        return approximate_spectral_radius(M.astype('float'), tol=tol, maxiter=20, restart=10)
    else:
        return np.absolute(eigs(M.astype('float'), k=1, return_eigenvectors=False, which='LM', tol=1e-04)[0])   # which='LM': largest magnitude; eigs / eigsh


def eps_convergence_linbp_parameterized(H, W,
                                        method='echo',
                                        alpha=0, beta=0, gamma=0,
                                        rho_W=None,
                                        X=None,
                                        pyamg=True):
    """Simplifies parameterization of noecho, echo, echo with compensation into one parameter
    """
    assert method in {'noecho', 'echo', 'comp'}
    echo = True
    compensation = False
    indices = None
    if X is not None:
        indices = to_explicit_list(X)
    if method == 'echo':
        None
    elif method == 'noecho':
        echo = False
    elif method == 'comp':
        compensation = True
    return eps_convergence_linbp(H, W_star(W, alpha=alpha, beta=beta, gamma=gamma, indices=indices),
                                 echo=echo,
                                 compensation=compensation,
                                 exponentiation=False,  # deprecated
                                 pyamg=pyamg
                                 )


def eps_convergence_linbp(Hc, W,
                          echo=False,
                          compensation=False,
                          exponentiation=False,         # exponentiation is deprecated
                          pyamg = True):
    """Calculates eps_convergence with which to multiply H so that LinBP (with or w/o echo) converges.
    Returns 0 if the entries of H are too small (std < SPECTRAL_TOLERANCE).
    Assumes symmetric W and symmetric H.

    Uses: degree_matrix, approx_spectral_radius

    Parameters
    ----------
    Hc : np.array
        Residual coupling matrix (all rows and colums sums to 0, derived from symmetric, doubly stochastic matrix)
    W : sparse matrix
        Sparse edge matrix (symmetric)
    echo : boolean (Default=False)
        True to include the echo cancellation term
    compensation : boolean (Default=False)
        If True, then calculates the exact compensation for echo H* (only works if echo=True)
        Only semantically correct if W is unweighted (TODO: extend with more general formula)
    exponentiation: Boolean (Default=False)
        Alternative version of centering a strong H potential (appears not to work as well and deprecated)
        DEPRECATED
    """
    if not check_centered_beliefs(Hc):
        warnings.warn("\nWarning from 'eps_convergence_linbp': Input is not a centered H matrix")
    if np.std(Hc) < SPECTRAL_TOLERANCE:
        return 0

    # -- Start with rho_W
    rho_W = approx_spectral_radius(csr_matrix(W, dtype='f'), pyamg=pyamg)  # needs to enforce float (not int)

    # -- Situation for standard linBP
    if not exponentiation:
        rho_H = approx_spectral_radius(np.array(Hc, dtype='f'), pyamg=pyamg)  # same here
        eps = 1. / rho_W / rho_H

        # -- If echo is used, then the above eps value is used as starting point
        if echo:
            Hc2 = Hc.dot(Hc)
            D = degree_matrix(W, undirected=True, squared=True)

            if not compensation:
                eps0 = eps / 1.9  # reason for echo with compensation. See "160716 - Spectral radius with echo and compensation.py"


                # function for which we need to determine the root: spectral radius minus 1
                def radius(eps):
                    return approx_spectral_radius(kron(Hc, W).dot(eps) - kron(Hc2, D).dot(eps ** 2), pyamg=pyamg) - 1

            # -- If compensation is used, then the formula is just more complicated. Also eps0 needs to be smaller
            else:
                eps0 = 0.5 / rho_H


                def radius(eps):
                    H_star = np.linalg.inv(np.identity(len(Hc)) - Hc2.dot(eps ** 2)).dot(Hc)
                    return approx_spectral_radius(kron(H_star, W).dot(eps) - kron(Hc.dot(H_star), D).dot(eps ** 2), pyamg=pyamg) - 1


            # http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html#scipy.optimize.newton
            eps = np.absolute(newton(radius, eps0, tol=1e-05, maxiter=100))


    # -- Situation for case of exponentiation (but no echo allowed!)
    # TODO: DEPRECATED
    else:
        def radius_Hc(eps):
            return approx_spectral_radius(centered_matrix_exponentiation(Hc, eps), pyamg=pyamg) - 1 / rho_W


        eps0 = 1
        eps = newton(radius_Hc, eps0, tol=1e-04, maxiter=100)

    return eps



def eps_convergence_directed_linbp(P, W,
                                   echo=False,
                                   originalVariant=True,
                                   pyamg=True,
                                   tol=1e-04):
    """Calculates eps_convergence with which to multiply row nad column-recentered potentials so that LinBP (with or w/o echo) converges.
    Returns 0 if the entries of Pc1 are too small (std < SPECTRAL_TOLERANCE).
    Potential is not centered and not row-normalized in contrast to symmetric version

    Uses: row_recentered_residual, approx_spectral_radius, degree_matrix

    Parameters
    ----------
    P: np.array
        Potential
    W: sparse matrix
        Sparse edge matrix of a directed graph
    echo : boolean (Default=False)
        True to include the echo term for LinBP
    originalVariant=True
        Corresponds to 'exponentiation=False option' option for undirected version. Not used

    """
    assert (P > 0).all()            # check that P is not residual, by coincidence,
    Pc1T = row_recentered_residual(P, paperVariant=originalVariant).transpose()
    Pc2 = row_recentered_residual(P.transpose(), paperVariant=originalVariant).transpose()
    WsT = W.transpose()

    if np.std(Pc1T) < SPECTRAL_TOLERANCE:
        return 0

    rho = approx_spectral_radius(kron(WsT, Pc1T) + kron(W, Pc2), pyamg=pyamg, tol=tol)
    eps = 1. / rho


    if echo:
        eps0 = eps
        D_in = degree_matrix(W, indegree=True, undirected=False, squared=True)
        D_out = degree_matrix(W, indegree=False, undirected=False, squared=True)
        Pstar1 = Pc1T.dot(Pc2)          # former error: 'Pc1.transpose() * Pc2' / even before error: 'Pc2.transpose() * Pc1'
        Pstar2 = Pc2.dot(Pc1T)

        def radius(eps):
            return approx_spectral_radius(
                (kron(WsT, Pc1T) + kron(W, Pc2)).dot(eps)
                - (kron(D_in, Pstar1) + kron(D_out, Pstar2)).dot(eps**2),
                pyamg=pyamg, tol=tol
                ) - 1
        eps = newton(radius, eps0, tol=tol, maxiter=100)

        if eps < 0:
            print("** eps negative in 1st iteration:", eps, ", original starting point:", eps0)
            eps = newton(radius, -eps, tol=1e-05, maxiter=100)
        if eps < 0:
            print("** eps negative in 2nd iteration:", eps)
            assert eps > 0, "** eps should be positive after second time"


    return eps

