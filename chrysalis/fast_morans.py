import numpy as np
from numba import njit


@njit
def moran_sparse(x, w_data, w_indices, w_indptr):
    n = len(x)
    x_mean = 0.0
    for i in range(n):
        x_mean += x[i]
    x_mean /= n

    num = 0.0
    denom = 0.0
    for i in range(n):
        xi = x[i] - x_mean
        for j_idx in range(w_indptr[i], w_indptr[i + 1]):
            j = w_indices[j_idx]
            w_ij = w_data[j_idx]
            num += w_ij * xi * (x[j] - x_mean)
        denom += xi ** 2
    return (n / np.sum(w_data)) * (num / denom)

@njit
def moran_sparse_matrix(X, w_data, w_indices, w_indptr):
    n, m = X.shape
    morans = np.empty(m)

    for k in range(m):
        x = np.empty(n)
        for i in range(n):
            x[i] = X[i, k]
        morans[k] = moran_sparse(x, w_data, w_indices, w_indptr)

    return morans
