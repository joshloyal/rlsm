import numpy as np


def dyad_cat_diff(x):
    n_nodes = x.shape[0]
    z = x.values.reshape(-1, 1)
    return ((z - z.T) == 0).astype(float)


def dyad_nominal_diff(x, standardize=False):
    n_nodes = x.shape[0]

    if standardize:
        x = (x - x.mean()) / x.std()

    z = x.values.reshape(-1, 1)
    return np.abs(z - z.T)
