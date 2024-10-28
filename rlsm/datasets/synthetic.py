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


def find_odds_ratio(dist_coef, distances, target):
    def target_func(rho):
        return np.exp(rho + dist_coef * distances).mean() - target

    return root_scalar(target_func, bracket=[-10, 10]).root


def generate_data(n_nodes=100, n_features=2, density=0.25, odds_ratio=2.,
                  mu=1, dist_coef=None, random_state=42):
    rng = check_random_state(random_state)
    key = PRNGKey(random_state)

    sigma_sr =  np.array([[1., 0.5],
                          [0.5, 1.]])
    sr = rng.multivariate_normal(
            np.zeros(2), sigma_sr, size=n_nodes)
    s, r = sr[:, 0].reshape(-1, 1), sr[:, 1].reshape(-1, 1)

    c = rng.choice([0, 1, 2], size=n_nodes)
    mu = np.array([[-mu, 0],
                   [mu, 0],
                   [0, mu]]) 
    Z = mu[c] + np.sqrt(0.1) * rng.randn(n_nodes, n_features)
    
    triu = np.triu_indices(n_nodes, k=1)
    distances = np.sqrt(pairwise_distance(Z)[triu])
    sr = adjacency_to_dyads(s + r.T, n_nodes)
    
    #recip_coef = rng.uniform(-1, 1) if recip_coef is None else recip_coef
    dist_coef = rng.uniform(-1, 1) if dist_coef is None else dist_coef

    recip_coef = rng.uniform(-1, 1) if odds_ratio is None else find_odds_ratio(dist_coef, distances, odds_ratio) 
    edge_coef = find_intercept(recip_coef, sr, distances, dist_coef, density)

    logits = to_logits(recip_coef, edge_coef + sr, distances, dist_coef)
    probas = to_probs(recip_coef, edge_coef + sr, distances, dist_coef)
    y_dyads = BivariateBernoulli(logits=logits).sample(key)

    Y = np.asarray(vec_to_adjacency(y_dyads))
    
    params = {
            'Z': Z,
            'c': c,
            's': s,
            'r': r,
            's_var': sigma_sr[0,0],
            'r_var': sigma_sr[1,1],
            'sr_corr': sigma_sr[0,1] / np.sqrt(sigma_sr[0,0] * sigma_sr[1,1]),
            'recip_coef': recip_coef,
            'dist_coef': dist_coef, 
            'probas': probas
    }

    return Y, params
