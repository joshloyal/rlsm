import numpy as np

from jax.random import PRNGKey
from scipy.special import expit
from scipy.optimize import root_scalar
from sklearn.utils import check_random_state

from ..distributions import BivariateBernoulli, to_logits, to_probs
from ..network_utils import adjacency_to_dyads, vec_to_adjacency


def pairwise_distance(U):
    U_norm_sq = np.sum(U ** 2, axis=1).reshape(-1, 1)
    dist_sq = U_norm_sq + U_norm_sq.T - 2 * U @ U.T
    return dist_sq


def find_intercept(recip_coef, ab, distances, dist_coef, target_density):
    def density_func(intercept):
        return to_probs(recip_coef, intercept + ab, distances, dist_coef).mean() - target_density

    return root_scalar(density_func, bracket=[-10, 10]).root


def generate_data(n_nodes=100, n_features=2, 
                  density=0.2, recip_coef=1, mu=2, dist_coef=1, random_state=42):
    rng = check_random_state(random_state)
    key = PRNGKey(random_state)

    sigma_ab =  np.array([[1., 0.5],
                          [0.5, 1.]])
    ab = rng.multivariate_normal(
            np.zeros(2), sigma_ab, size=n_nodes)
    a, b = ab[:, 0].reshape(-1, 1), ab[:, 1].reshape(-1, 1)

    z = rng.choice([0,1,2], size=n_nodes)
    mu = np.array([[-mu, 0],
                   [mu, 0],
                   [0, mu]]) 
    U = mu[z] + np.sqrt(0.1) * rng.randn(n_nodes, n_features)
    
    triu = np.triu_indices(n_nodes, k=1)
    distances = np.sqrt(pairwise_distance(U)[triu])
    ab = adjacency_to_dyads(a + b.T, n_nodes)

    edge_coef = find_intercept(recip_coef, ab, distances, dist_coef, density)
    logits = to_logits(recip_coef, edge_coef + ab, distances, dist_coef)
    probas = to_probs(recip_coef, edge_coef + ab, distances, dist_coef)
    y_dyads = BivariateBernoulli(logits=logits).sample(key)

    Y = np.asarray(vec_to_adjacency(y_dyads))

    return Y, U, z, edge_coef + a.ravel(), edge_coef + b.ravel(), probas
