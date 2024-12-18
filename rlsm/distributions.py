import numpy as np
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
import jax.random
from jax import lax
from jax.scipy.special import expit
from jax.nn import softmax
from .network_utils import dyads_to_vec


def to_probs(recip_coef, sr, dist, dist_coef, reciprocity_type='distance'):
    probas = softmax(
        to_logits(recip_coef, sr, dist, dist_coef, reciprocity_type), axis=-1)
    return dyads_to_vec(
            jnp.c_[probas[:, 0] + probas[:, 1], probas[:, 0] + probas[:, 2]])  


def to_logits(recip_coef, sr, dist, dist_coef, reciprocity_type='distance'):
    if reciprocity_type == 'distance':
        mu11 = sr[:, 0] + sr[:, 1] + recip_coef + (dist_coef - 2) *  dist
    else:
        mu11 = sr[:, 0] + sr[:, 1] + recip_coef - 2 * dist

    mu10 = sr[:, 0] - dist
    mu01 = sr[:, 1] - dist
    mu00 = jnp.zeros_like(mu01)
    return jnp.c_[mu11, mu10, mu01, mu00]
    

class BivariateBernoulli(dist.MultinomialLogits):
    def sample(self, key, sample_shape=()):
        y_obs = super(BivariateBernoulli, self).sample(key, sample_shape)
        dyad_map = jnp.array([[1., 1.],
                              [1., 0.],
                              [0., 1.],
                              [0., 0.]])
        return dyads_to_vec(
                jnp.take(dyad_map, jnp.argmax(y_obs, axis=1), axis=0)) 
