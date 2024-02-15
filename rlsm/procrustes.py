import numpy as np
import numpy.linalg as linalg
from scipy.linalg import orthogonal_procrustes


def flatten_array(X):
    return X.reshape(np.prod(X.shape[:-1]), -1)


def static_procrustes_rotation(X, Y):
    """Rotate Y to match X"""
    R, _ = orthogonal_procrustes(Y, X)
    return Y @ R, R


def longitudinal_procrustes_rotation(X_ref, X):
    """A single procrustes transformation applied across time."""
    n_time_steps, n_nodes = X.shape[:-1]

    X_ref = flatten_array(X_ref)
    X = flatten_array(X)
    X, R = static_procrustes_rotation(X_ref, X)
    return X.reshape(n_time_steps, n_nodes, -1), R
