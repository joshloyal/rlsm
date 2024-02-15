import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from joblib import Parallel, delayed
from jax.scipy.special import expit, logsumexp
from scipy.linalg import orthogonal_procrustes
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density
from numpyro.diagnostics import print_summary as numpyro_print_summary
from numpyro.distributions import constraints
from sklearn.metrics import roc_auc_score

from jax import jit, random, vmap
from sklearn.utils import check_random_state

from .network_utils import (
    nondiag_indices, adjacency_to_vec, adjacency_to_dyads,
    adjacency_to_dyads_multinomial)
from .distributions import BivariateBernoulli, to_probs, to_logits
from .plots import plot_model
from .mcmc_utils import condition


def init_to_uniform(low=-2., high=2., shape=()):
    def init_fn(rng_key):
        return random.uniform(rng_key, shape=shape, minval=low, maxval=high)
    return init_fn


def init_to_normal(shape=()):
    def init_fn(rng_key):
        return random.normal(rng_key, shape=shape)
    return init_fn


def posterior_predictive(model, rng_key, samples, stat_fun, *model_args,
                         **model_kwargs):
    model = handlers.seed(condition(model, samples), rng_key)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return stat_fun(model_trace["Y"]["value"])


def predict_proba(model, rng_key, samples, *model_args, **model_kwargs):
    model = handlers.seed(condition(model, samples), rng_key)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return model_trace["probas"]["value"]


def log_likelihood(model, rng_key, samples, *model_args, **model_kwargs):
    model = handlers.seed(condition(model, samples), rng_key)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    obs_node = model_trace["Y"]
    return obs_node["fn"].log_prob(obs_node["value"])


def calculate_posterior_predictive(mcmc, stat_fun, random_state, *model_args,
                                   **model_kwargs):
    rng_key = random.PRNGKey(random_state)

    samples = mcmc.get_samples()
    n_samples  = samples['U'].shape[0]
    vmap_args = (samples, random.split(rng_key, n_samples))

    return vmap(
        lambda samples, rng_key : posterior_predictive(
            model, rng_key, samples, stat_fun,
            *model_args, **model_kwargs)
    )(*vmap_args)


def print_summary(samples, divergences, prob=0.9):
    #fields = ['recip_coef', 'dist_coef',
    #          'a_sigma', 'b_sigma', 'ab_sigma', 'u_sigma']
    fields = ['recip_coef', 'a_sigma', 'b_sigma', 'ab_sigma', 'u_sigma']
    samples = {k: v for k, v in samples.items() if
        k in fields and k in samples.keys()}
    samples = jax.tree_map(lambda x : jnp.expand_dims(x, axis=0), samples)
    numpyro_print_summary(samples, prob=prob)
    if divergences is not None:
        print(f"Number of divergences: {divergences}")


def pairwise_distance(U):
    U_norm_sq = jnp.sum(U ** 2, axis=1).reshape(-1, 1)
    dist_sq = U_norm_sq + U_norm_sq.T - 2 * U @ U.T
    return dist_sq

    
def rlsm(Y, n_nodes, n_features=2, include_reciprocity=True, is_predictive=False):
    
    # cholesky decomposition of sender/receiver covariance
    omega_ab = numpyro.sample("omega_ab",
        dist.LKJCholesky(2, concentration=0.5))
    gamma_ab = numpyro.sample("gamma_ab",
        dist.HalfCauchy(scale=jnp.ones(2)))

    ## centered sender/receiver parameters
    L_ab = jnp.diag(gamma_ab) @ omega_ab
    Sigma_ab = L_ab @ L_ab.T
    numpyro.deterministic('a_sigma', Sigma_ab[0, 0])
    numpyro.deterministic('b_sigma', Sigma_ab[1, 1])
    numpyro.deterministic('ab_sigma', Sigma_ab[0, 1])
    z_ab = numpyro.sample("z_ab",
        dist.Normal(jnp.zeros((2, n_nodes)), jnp.ones((2, n_nodes))))
    
    # intercept
    edge_coef = numpyro.sample('edge_coef', dist.Normal(0, 10.))

    # generate sender/receiver effects
    ab = edge_coef + L_ab @ z_ab
    a = numpyro.deterministic("a", ab[0]).reshape(-1, 1)
    b = numpyro.deterministic("b", ab[1]).reshape(-1, 1)

    # latent positions
    u_sigma = numpyro.sample('u_sigma',dist.Gamma(1., 0.5))
    if n_features is not None and n_features > 0 :
        Z = numpyro.sample('Z',
            dist.Normal(jnp.zeros((n_nodes, n_features)),
                        jnp.ones((n_nodes, n_features))))
        U = numpyro.deterministic('U', u_sigma * Z)

 
    # reciprocity
    if include_reciprocity:
        recip_coef = numpyro.sample("recip_coef", dist.Normal(0., 10.))  
        dist_coef = 1.
    else:
        recip_coef = numpyro.deterministic("recip_coef", 0.)
        dist_coef = 0.
    
    # coefficient in front of distance in reciprocity term
    #dist_coef = numpyro.sample("dist_coef", dist.Normal(0., 10.))
    
    # nodewise reciprocity
    #gamma_sigma = numpyro.sample("gamma_sigma", dist.Gamma(1., 0.5))
    #z = numpyro.sample("z_gamma", dist.Normal(
    #    jnp.zeros(n_nodes), jnp.ones(n_nodes)))
    #gamma = numpyro.deterministic("gamma", gamma_sigma * z)
    #gamma = gamma.reshape(-1, 1)

    triu = jnp.triu_indices(n_nodes, k=1)
    distances = jnp.sqrt(pairwise_distance(U)[triu])
    #node_rep = (gamma + gamma.T)[triu]
    ab = adjacency_to_dyads(a + b.T, n_nodes)

    # likelihood
    with numpyro.handlers.condition(data={"Y": Y}):
        #logits = to_logits(edge_coef, recip_coef, dist_coef, ab, distances)
        logits = to_logits(recip_coef, dist_coef, ab, distances)
        y = numpyro.sample("Y", BivariateBernoulli(logits=logits))

        if is_predictive:
            probas = numpyro.deterministic('probas', 
                    #to_probs(edge_coef, recip_coef, dist_coef, ab, distances))
                    to_probs(recip_coef, dist_coef, ab, distances))


class ReciprocityLSM(object):
    def __init__(self,
                 n_features=None,
                 include_reciprocity=True,
                 random_state=42):
        self.n_features = n_features
        self.include_reciprocity = include_reciprocity
        self.random_state = random_state

    @property
    def model_args_(self):
        n_nodes = self.samples_['U'].shape[1]
        return (None, n_nodes, self.n_features, self.include_reciprocity)

    @property
    def model_kwargs_(self):
        return {'is_predictive': True}

    def sample(self, Y, 
            n_warmup=1000, n_samples=1000, adapt_delta=0.8, n_iter=10000):
        #numpyro.enable_x64()

        n_nodes = Y.shape[0]
        self.Y_fit_ = Y.copy()
        y = adjacency_to_dyads_multinomial(Y, n_nodes)

        # run mcmc sampler
        rng_key = random.PRNGKey(self.random_state)
        model_args = (y, n_nodes, self.n_features, self.include_reciprocity, False)

        mcmc = MCMC(NUTS(rlsm), num_warmup=n_warmup, num_samples=n_samples,
                    num_chains=1)
        mcmc.run(rng_key, *model_args)
        self.diverging_ = jnp.sum(mcmc.get_extra_fields()['diverging'])

        # extract/process samples
        self.samples_ = mcmc.get_samples()
        
        # calculate log density
        self.logp_ = vmap(
            lambda sample : log_density(
                rlsm, model_args=(y, n_nodes, self.n_features), 
                model_kwargs={'is_predictive': False},
                params=sample)[0])(self.samples_)
        self.map_idx_ = np.argmax(self.logp_)
        self.samples_ = jax.tree_map(lambda x : np.array(x), self.samples_)
        
        U_ref = self.samples_['U'][self.map_idx_]
        for idx in range(self.samples_['U'].shape[0]):
            R, _ = orthogonal_procrustes(self.samples_['U'][idx], U_ref)
            self.samples_['U'][idx] = self.samples_['U'][idx] @ R
        
        self.a_ = self.samples_['a'].mean(axis=0)
        self.b_ = self.samples_['b'].mean(axis=0)
        #self.edge_coef_ = self.samples_['edge_coef'].mean()
        self.recip_coef_ = self.samples_['recip_coef'].mean()
        #self.dist_coef_ = self.samples_['dist_coef'].mean()
        self.U_ = self.samples_['U'].mean(axis=0)

        self.probas_ = self.predict_proba()
        self.auc_ = roc_auc_score(adjacency_to_vec(self.Y_fit_), self.probas_)
        
        return self

    def posterior_predictive(self, stat_fun, random_state=42):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))

        return np.asarray(vmap(
            lambda samples, rng_key : posterior_predictive(
                rlsm, rng_key, samples, stat_fun,
                *self.model_args_, **self.model_kwargs_)
        )(*vmap_args))

    def predict_proba(self, random_state=42):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))

        return np.asarray(vmap(
            lambda samples, rng_key : predict_proba(
                rlsm, rng_key, samples,
                *self.model_args_, **self.model_kwargs_)
        )(*vmap_args).mean(axis=0))

    def print_summary(self, proba=0.9):
        print_summary(self.samples_, self.diverging_, prob=proba)

    def plot(self, Y_obs=None, **fig_kwargs):
        Y_obs = Y_obs if Y_obs is not None else self.Y_fit_
        return plot_model(self, Y_obs, **fig_kwargs)
    
    def waic(self, random_state=0):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        n_nodes = self.samples_['U'].shape[1]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))
        
        y = adjacency_to_dyads_multinomial(self.Y_fit_, n_nodes)
        model_args = (y, n_nodes, self.n_features, self.include_reciprocity)
        
        loglik = vmap(
            lambda samples, rng_key : log_likelihood(
                rlsm, rng_key, samples,
                *model_args, **self.model_kwargs_))(*vmap_args)

        lppd = (logsumexp(loglik, axis=0) - jnp.log(n_samples)).sum()
        p_waic = loglik.var(axis=0).sum()
        return float(-2 * (lppd - p_waic))
