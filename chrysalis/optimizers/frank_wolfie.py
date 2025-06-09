import numpy as np
from sklearn.utils.extmath import squared_norm


def frank_wolfe_single_row(x, W, max_iter=100, tol=1e-4):
    k = W.shape[0]
    h = np.ones(k) / k  # Start on the simplex

    for _ in range(max_iter):
        grad = -2 * (x - h @ W) @ W.T  # Gradient of squared error
        s = np.zeros_like(h)
        s[np.argmin(grad)] = 1.0  # Linear minimization over simplex

        step_size = 2 / (_ + 2)  # Can be tuned or line searched
        h_new = (1 - step_size) * h + step_size * s

        if np.linalg.norm(h_new - h) < tol:
            break
        h = h_new

    return h


def frank_wolfe_fit_transform(X, A, B, archetypes, *, max_iter, tol, verbose, **kwargs):
    loss_list = [
        squared_norm(X - A @ archetypes),
    ]

    for i in range(1, max_iter + 1):
        # Update A row-by-row using Frank-Wolfe
        A = np.vstack([
            frank_wolfe_single_row(x_i, archetypes, **kwargs)
            for x_i in X
        ])

        # Update B using pseudo-inverse and FW (same idea, different side)
        # You may keep NNLS or just do least-squares here
        X_tilde = np.linalg.pinv(A) @ X
        B = np.vstack([
            frank_wolfe_single_row(x_j, X, **kwargs)
            for x_j in X_tilde
        ])

        archetypes = B @ X  # Reconstruct archetypes

        rss = squared_norm(X - A @ archetypes)
        convergence = abs(loss_list[-1] - rss) < tol
        loss_list.append(rss)

        if verbose and i % 10 == 0:
            verbose_print_rss(max_iter, rss, i)
        if convergence:
            break

    return A, B, archetypes, i, loss_list, convergence


def frank_wolfe_transform(X, archetypes, *, max_iter, tol, **kwargs):
    # todo: write this
    pass
