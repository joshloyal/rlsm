import networkx as nx
import numpy as np
import scipy.linalg as linalg

from scipy.sparse import csgraph
from sklearn.manifold import MDS


def shortest_path_dissimilarity(Y, unweighted=True):
    """Calculate the shortest-path dissimilarty of a static graph."""
    dist = csgraph.shortest_path(Y, directed=False, unweighted=unweighted)

    # impute unconnected components with the largest distance plus 1
    inf_mask = np.isinf(dist)
    dist[inf_mask] = np.max(dist[~inf_mask]) + 1

    return dist


def initialize_mds(Y, n_features=2, random_state=None):
    """Generalized Multi-Dimension Scaling (Sarkar and Moore, 2005)."""

    n_nodes, _ = Y.shape

    # calculate shortest-path dissimilarity for each time step
    D = shortest_path_dissimilarity(Y, unweighted=True)

    # compute latent positions based on MDS
    X = MDS(dissimilarity='precomputed',
            n_components=n_features,
            random_state=random_state).fit_transform(D)
    
    # standardize
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    return X
