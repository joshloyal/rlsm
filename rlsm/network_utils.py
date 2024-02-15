import numpy as np
import jax.numpy as jnp


def nondiag_indices(n):
    tri = jnp.tril_indices(n, k=-1)
    tru = jnp.triu_indices(n, k=1)
    nondiag = jnp.hstack((tru, tri)).T

    return (nondiag[:, 0], nondiag[:, 1])


def adjacency_to_dyads(Y, n):
    triu = jnp.triu_indices(n, k=1)
    return jnp.hstack((Y.T[triu].reshape(-1, 1), Y[triu].reshape(-1, 1)))


def adjacency_to_dyads_multinomial(Y, n):
    y = adjacency_to_dyads(Y, n)
    return jnp.c_[y[:, 0] * y[:, 1],
           y[:, 0] * (1 - y[:, 1]),
           (1 - y[:, 0]) * y[:, 1],
           (1 - y[:, 0]) * (1 - y[:, 1])]


def adjacency_to_vec(Y):
    n_nodes = Y.shape[0]
    n_dyads = int(n_nodes * (n_nodes - 1))
    nondiag = nondiag_indices(n_nodes)
    return jnp.asarray(Y[nondiag])


def vec_to_adjacency(y, include_nan=False):
    n_dyads = y.shape[0]
    n = round(0.5 * (1 + jnp.sqrt(1 + 4 * n_dyads)))
    Y = jnp.zeros((n, n))
    Y = Y.at[nondiag_indices(n)].set(y)
    if include_nan:
        Y = Y.at[jnp.diag_indices(n)].set(np.nan)
    return Y


def dyads_to_adjacency(y, include_nan=False):
    n_dyads = y.shape[0]
    n = round((np.sqrt(1 + 8 * n_dyads) - 1) / 2) + 1
    Y = jnp.zeros((n, n))
    Y = Y.T.at[jnp.triu_indices(n, k=1)].set(y[:, 0]).T
    Y = Y.at[jnp.triu_indices(n, k=1)].set(y[:, 1])
    if include_nan:
        Y = Y.at[jnp.diag_indices(n)].set(np.nan)
    return Y


def multinomial_to_adjacency(y, include_nan=False):
    dyad_map = jnp.array([[1., 1.],
                          [0., 1.],
                          [1., 0.],
                          [0., 0.]])
    y_dyads = dyad_map[jnp.argmax(y, axis=1)]
    return dyads_to_adjacency(y_dyads, include_nan=include_nan)


def dyads_to_vec(y):
    return adjacency_to_vec(dyads_to_adjacency(y))
