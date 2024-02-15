import numpy as np
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
import jax.random
from jax import lax
from jax.scipy.special import expit
from jax.nn import softmax
from .network_utils import dyads_to_vec


def to_probs(recip_coef, dist_coef, ab, dist):
    #mu10 = jnp.exp(edge_coef + ab[:, 0] - dist)
    #mu01 = jnp.exp(edge_coef + ab[:, 1] - dist)
    ##mu11 = jnp.exp(2 * edge_coef + ab[:, 0] + ab[:, 1] + recip_coef - dist)
    #mu11 = jnp.exp(2 * edge_coef + ab[:, 0] + ab[:, 1] + recip_coef + (dist_coef - 2) * dist)
    #p00 = 1 / (1 + mu10 + mu01 + mu11)
    #p10 = p00 * mu10
    #p01 = p00 * mu01
    #p11 = p00 * mu11
    #return dyads_to_vec(jnp.c_[p11 + p10, p11 + p01])
    
    probas = softmax(
            to_logits(recip_coef, dist_coef, ab, dist), axis=-1)
    return dyads_to_vec(
            jnp.c_[probas[:, 0] + probas[:, 1], probas[:, 0] + probas[:, 2]])  


def to_logits(recip_coef, dist_coef, ab, dist):
    #mu11 = 2 * edge_coef + ab[:, 0] + ab[:, 1] + recip_coef - dist
    mu11 = ab[:, 0] + ab[:, 1] + recip_coef + (dist_coef - 2) * dist
    #mu11 = ab[:, 0] + ab[:, 1] + recip_coef - (dist_coef - 2) * dist
    #mu11 = ab[:, 0] + ab[:, 1] + recip_coef 
    mu10 = ab[:, 0] - dist
    mu01 = ab[:, 1] - dist
    mu00 = jnp.zeros_like(mu01)
    return jnp.c_[mu11, mu10, mu01, mu00]
    

#class BivariateBernoulli(dist.MultinomialProbs):
#    def sample(self, key, sample_shape=()):
#        y_obs = super(BivariateBernoulli, self).sample(key, sample_shape)
#        dyad_map = jnp.array([[1., 1.],
#                              [1., 0.],
#                              [0., 1.],
#                              [0., 0.]])
#        return dyads_to_vec(jnp.take(dyad_map, jnp.argmax(y_obs, axis=1), axis=0))

class BivariateBernoulli(dist.MultinomialLogits):
    def sample(self, key, sample_shape=()):
        y_obs = super(BivariateBernoulli, self).sample(key, sample_shape)
        dyad_map = jnp.array([[1., 1.],
                              [1., 0.],
                              [0., 1.],
                              [0., 0.]])
        return dyads_to_vec(jnp.take(dyad_map, jnp.argmax(y_obs, axis=1), axis=0)) 
