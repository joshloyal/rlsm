import numpy as np

from jax.random import PRNGKey
from scipy.special import expit
from sklearn.utils import check_random_state

from ..distributions import BivariateBernoulli, to_logits
from ..network_utils import adjacency_to_dyads, vec_to_adjacency


def pairwise_distance(U):
    U_norm_sq = np.sum(U ** 2, axis=1).reshape(-1, 1)
    dist_sq = U_norm_sq + U_norm_sq.T - 2 * U @ U.T
    return dist_sq


def generate_data(n_nodes=100, n_features=2, 
                  edge_coef=-1, recip_coef=1, mu=2,
                  dist_coef=1., random_state=42):
    rng = check_random_state(random_state)
    key = PRNGKey(random_state)

    sigma_ab =  0.1 * np.array([[1., 0.5],
                                [0.5, 1.]])
    ab = edge_coef + rng.multivariate_normal(np.zeros(2), sigma_ab, size=n_nodes)
    a, b = ab[:, 0].reshape(-1, 1), ab[:, 1].reshape(-1, 1)

    z = rng.choice([0,1], size=n_nodes)
    mu = np.array([[-mu, 0],
                   [mu, 0]]) 
    U = mu[z] + np.sqrt(0.1) * rng.randn(n_nodes, n_features)
    
    #tau = 0.2
    #gamma = tau * rng.randn(n_nodes, 1)

    triu = np.triu_indices(n_nodes, k=1)
    distances = np.sqrt(pairwise_distance(U)[triu])
    #node_rep = (gamma + gamma.T)[triu]
    ab = adjacency_to_dyads(a + b.T, n_nodes)

    logits = to_logits(recip_coef, dist_coef, ab, distances)
    y_dyads = BivariateBernoulli(logits=logits).sample(key)

    Y = np.asarray(vec_to_adjacency(y_dyads))

    return Y, U, z, a.ravel(), b.ravel()
