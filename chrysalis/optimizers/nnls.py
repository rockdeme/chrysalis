import scipy
import numpy as np
from sklearn.utils.extmath import squared_norm


def verbose_print_rss(max_iter, rss, i):
    print(f"Iteration {i}/{max_iter}: RSS = {rss}")


def nnls(B, A, max_iter_optimizer=100, const=100.0):
    """
    Non-negative Least Squares optimization.
    B = A @ X
    :param B: np.ndarray The matrix to decompose.
    :param A: np.ndarray The matrix to decompose with.
    :param max_iter_optimizer:
    :param const:
    :return: A: np.ndarray The coefficients matrix.
    """
    B = np.hstack([B, const * np.ones((B.shape[0], 1))])
    A = np.hstack([A, const * np.ones((A.shape[0], 1))])

    X = np.zeros((B.shape[0], A.shape[0]))
    for i in range(B.shape[0]):
        X[i, :] = scipy.optimize.nnls(A.T, B[i, :], maxiter=max_iter_optimizer)[0]

    X = np.maximum(X, 1e-8)
    X = X / np.sum(X, axis=1)[:, None]

    return X


def nnls_transform(X, archetypes, *, max_iter, tol, **kwargs):
    return nnls(X, archetypes, **kwargs)


def nnls_fit_transform(X, A, B, archetypes, *, max_iter, tol, verbose, **kwargs):
    loss_list = [
        squared_norm(X - A @ archetypes),
    ]
    for i in range(1, max_iter + 1):
        A = nnls(X, archetypes, **kwargs)
        B = nnls(np.linalg.pinv(A) @ X, X, **kwargs)
        archetypes = np.matmul(B, X, out=archetypes)

        rss = squared_norm(X - A @ archetypes)
        convergence = abs(loss_list[-1] - rss) < tol
        loss_list.append(rss)
        if verbose and i % 10 == 0:  # Verbose mode (print RSS)
            verbose_print_rss(max_iter, rss, i)
        if convergence:
            break
    return A, B, archetypes, i, loss_list, convergence
