import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt


def return_cell_types(adata, annot_col):
    # return unique cell types from adata obs
    assert annot_col in adata.obs.columns

    # generate per cell type simulation attributes
    cell_types = np.unique(adata.obs[annot_col])
    n_cell_types = len(cell_types)

    assert n_cell_types > 1
    return cell_types, n_cell_types


def generate_grid(n):
    if n is None:
        n = (50, 50)
    n1, n2 = n
    x = np.linspace(0, 100, n1)[:, None]  # spatial dimensions
    y = np.linspace(0, 100, n2)[:, None]  # spatial dimensions

    # make cartesian grid out of each dimension x and y
    return pm.math.cartesian(x, y), x, y


def kernel(x, y, l=1.0, eta=1.0):
    """
    Isotropic squared exponential kernel. Computes a covariance matrix from points in x and y.
    :param x: Array of m points (m x d).
    :param y: Array of n points (n x d).
    :param l:
    :param eta:
    :return: Covariance matrix (m x n).
    """
    # squared euclidean distance of each point
    sqdist = np.sum(x ** 2, 1).reshape(-1, 1) + np.sum(y ** 2, 1) - 2 * np.dot(x, y.T)
    sqdist = eta ** 2 * np.exp(-0.5 / l ** 2 * sqdist)
    return sqdist


def random_gaussian(spatial_locs, x, y, n_variables=3, eta_true=1.5, l1_true=(8, 10, 15), l2_true=(8, 10, 15)):
    """
    :param spatial_locs:
    :param x: coordinates
    :param y: coordinates
    :param n_variables: zones
    :param eta_true: variance, defines overlapping
    :param l1_true: bw parameter
    :param l2_true: bw parameter
    :return:
    """

    k = [np.kron(kernel(x, y, l=l1_true[i], eta=eta_true),
                 kernel(x, y, l=l2_true[i], eta=eta_true)) for i in range(n_variables)]

    # samples from GP
    mean = np.zeros(spatial_locs.shape[0])
    gaus_true = [np.random.multivariate_normal(mean, 2 * k[i]) for i in range(n_variables)]
    gaus_true = np.stack(gaus_true).T
    # softmax transform
    if n_variables > 1:
        n_true = (np.exp(gaus_true).T / np.exp(gaus_true).sum(axis=1)).T
    else:  # handle the case when n_variables is 1
        max_gaus = np.max(gaus_true)
        exp_gaus = np.exp(gaus_true - max_gaus)
        n_true = (exp_gaus / np.sum(exp_gaus))

    return n_true


def sample_gp(locations, x, y, n_tissue_zones=3, l1_tissue_zones=(8, 10, 15), n_tissue_zones_unfirom=0,
              l1_tissue_zones_uniform=None, zone_names=None, col_name_prefix='tissue_zone', eta_true=1.5):
    """
    DEPRECATED

    :param locations:
    :param x:
    :param y:
    :param n_tissue_zones:
    :param l1_tissue_zones:
    :param n_tissue_zones_unfirom:
    :param l1_tissue_zones_uniform:
    :param zone_names:
    :return:
    """
    if zone_names is None:
        zone_names = [f'{col_name_prefix}_{f}' for f in range(len(l1_tissue_zones))]
    # Sample abundances with GP
    sparse_abundances = random_gaussian(locations, x=x, y=y, n_variables=n_tissue_zones,
                                        eta_true=eta_true,
                                        l1_true=l1_tissue_zones,
                                        l2_true=l1_tissue_zones)
    abundances = sparse_abundances / sparse_abundances.max(0)
    abundances[abundances < 0.1] = 0

    if n_tissue_zones_unfirom > 0:
        uniform_abundances = random_gaussian(locations, x=x, y=y, n_variables=n_tissue_zones_unfirom,
                                             eta_true=0.5,
                                             l1_true=l1_tissue_zones_uniform,
                                             l2_true=l1_tissue_zones_uniform)
        uniform_abundances = uniform_abundances / uniform_abundances.max(0)
        uniform_abundances[uniform_abundances < 0.1] = 0

        abundances = np.concatenate([sparse_abundances, uniform_abundances], axis=1)
    indexes = [f'location_{i}' for i in range(abundances.shape[0])]
    df = pd.DataFrame(abundances, index=indexes, columns=zone_names)
    return df


def plot_spatial(values, n=(50, 50), nrows=5, names=['cell type'], vmin=0, vmax=1):
    n_cell_types = values.shape[1]
    n1, n2 = n
    ncols = int(np.ceil((n_cell_types + 1) / nrows))
    for ct in range(n_cell_types):
        plt.subplot(nrows, ncols, ct + 1)
        plt.imshow(values[:, ct].reshape(n1, n2).T,
                   cmap=plt.cm.get_cmap('magma'),
                   vmin=vmin, vmax=vmax)
        plt.colorbar(fraction=0.047)
        if len(names) > 1:
            plt.title(names[ct])
        else:
            plt.title(f'{names[0]} {ct + 1}')

    plt.subplot(nrows, ncols, n_cell_types + 1)
    plt.imshow(values.sum(axis=1).reshape(n1, n2).T,
               cmap=plt.cm.get_cmap('Greys'))
    plt.colorbar(fraction=0.047)
    plt.title('Total')


def reaction_diffusion(zoom=(50, 50), size=100, t=16.0, dt=0.001, a=0.0001, b=0.005, show_generation=False):
    """

    Implemented from: "https://ipython-books.github.io/
    124-simulating-a-partial-differential-equation-reaction-diffusion-systems-and-turing-patterns/"

    :param zoom:
    :param size: size of the 2D grid
    :param t: total time
    :param dt: time step
    :param a: 0.0001 - 0.001  - granularity decreasing
    :param b: 0.0005 - 0.005 - gradient steepness increasing
    :param show_generation:
    :return:
    """

    k = -0.005
    tau = 0.1
    dx = 2. / size  # space step
    n = int(t / dt)  # number of iterations

    u = np.random.rand(size, size)
    v = np.random.rand(size, size)

    def laplacian(z):
        z_top = z[0:-2, 1:-1]
        z_left = z[1:-1, 0:-2]
        z_bottom = z[2:, 1:-1]
        z_right = z[1:-1, 2:]
        z_center = z[1:-1, 1:-1]
        return (z_top + z_left + z_bottom + z_right - 4 * z_center) / dx**2

    def show_patterns(u, ax=None):
        ax.imshow(u, cmap="viridis", interpolation=None, extent=[-1, 1, -1, 1])
        ax.set_axis_off()

    if show_generation:
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        step_plot = n // 9
    # We simulate the PDE with the finite difference method.
    for i in range(n):
        # We compute the Laplacian of u and v.
        delta_u = laplacian(u)
        delta_v = laplacian(v)
        # We take the values of u and v inside the grid.
        u_c = u[1:-1, 1:-1]
        v_c = v[1:-1, 1:-1]
        # We update the variables.
        u[1:-1, 1:-1], v[1:-1, 1:-1] = \
            u_c + dt * (a * delta_u + u_c - u_c ** 3 - v_c + k),\
            v_c + dt * (b * delta_v + u_c - v_c) / tau
        # Neumann conditions: derivatives at the edge are null.
        for z in (u, v):
            z[0, :] = z[1, :]
            z[-1, :] = z[-2, :]
            z[:, 0] = z[:, 1]
            z[:, -1] = z[:, -2]

        # We plot the state of the system at 9 different times.
        if show_generation:
            if i % step_plot == 0 and i < 9 * step_plot:
                ax = axes.flat[i // step_plot]
                show_patterns(u, ax=ax)
                ax.set_title(f'$t={i * dt:.2f}$')

    if show_generation:
        plt.show()

    u = u[:zoom[0], :zoom[1]]
    u = (u - np.min(u)) / (np.max(u) - np.min(u))
    # rescaled = resize(U, (ls, ls), order=1)

    if show_generation:
        plt.imshow(u)
        plt.colorbar()
        plt.show()

    return u


def random_reaction_diffusion(grid_size=(50, 50), n_variables=3, param_a=(0.0004, 0.0006, 0.0008)):
    """

    :param grid_size:
    :param n_variables:
    :param param_a:
    :return:
    """
    # samples from reaction-diffusion

    r_d_true = []
    for n, a in zip(range(n_variables), param_a):
        print(f'a = {a}')
        r_d =  reaction_diffusion(zoom=grid_size, size=100, t=8, a=a)
        r_d = r_d.ravel()
        r_d_true.append(r_d)
    r_d_true = np.stack(r_d_true).T

    return r_d_true


def sample_model(locations, x, y, n_tissue_zones=3, pattern_type='gp', l1_tissue_zones=(8, 10, 15), a=(4, 6, 8),
                 n_tissue_zones_unfirom=0, l1_tissue_zones_uniform=None, zone_names=None,
                 col_name_prefix='tissue_zone', eta_true=1.5, eta_true_uniform=0.5):
    """

    :param locations:
    :param x:
    :param y:
    :param n_tissue_zones:
    :param l1_tissue_zones:
    :param n_tissue_zones_unfirom:
    :param l1_tissue_zones_uniform:
    :param zone_names:
    :return:
    """
    if zone_names is None:
        zone_names = [f'{col_name_prefix}_{f}' for f in range(len(l1_tissue_zones))]
    if pattern_type == 'gp':
        # Sample abundances with GP
        sparse_abundances = random_gaussian(locations, x=x, y=y, n_variables=n_tissue_zones,
                                            eta_true=eta_true,
                                            l1_true=l1_tissue_zones,
                                            l2_true=l1_tissue_zones)
    elif pattern_type == 'r_d':
        # sample abundances with Reaction-Diffusion
        grid_x = len(x)
        grid_y = len(y)
        a = a / 10000
        sparse_abundances = random_reaction_diffusion(grid_size=(grid_x, grid_y), n_variables=n_tissue_zones,
                                                      param_a=a)
    else:
        raise Exception("pattern_type parameter should be one of the following: ['gp', 'r_d']")

    sparse_abundances = sparse_abundances / sparse_abundances.max(0)
    sparse_abundances[sparse_abundances < 0.1] = 0

    if n_tissue_zones_unfirom > 0:
        uniform_abundances = random_gaussian(locations, x=x, y=y, n_variables=n_tissue_zones_unfirom,
                                             eta_true=eta_true_uniform,
                                             l1_true=l1_tissue_zones_uniform,
                                             l2_true=l1_tissue_zones_uniform)
        uniform_abundances = uniform_abundances / uniform_abundances.max(0)
        uniform_abundances[uniform_abundances < 0.1] = 0

        uniform_zone_names = [f'uniform_{col_name_prefix}_{f}' for f in range(len(l1_tissue_zones_uniform))]
        zone_names = zone_names + uniform_zone_names
        abundances = np.concatenate([sparse_abundances, uniform_abundances], axis=1)
    else:  # if we have no uniform tissue zones
        abundances = sparse_abundances
    indexes = [f'location_{i}' for i in range(abundances.shape[0])]
    df = pd.DataFrame(abundances, index=indexes, columns=zone_names)
    return df
