import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import arviz as az

from collections import OrderedDict
from jax.scipy.special import expit, logsumexp, logit
from scipy.linalg import orthogonal_procrustes
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density
from numpyro.diagnostics import print_summary as numpyro_print_summary
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from jax import jit, random, vmap
from sklearn.utils import check_random_state

from .network_utils import (
    nondiag_indices, adjacency_to_vec, adjacency_to_dyads,
    adjacency_to_dyads_multinomial)
from .distributions import BivariateBernoulli, to_probs, to_logits
from .plots import plot_model
from .mcmc_utils import condition
from .initialize import initialize_mds


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
    n_samples  = samples['Z'].shape[0]
    vmap_args = (samples, random.split(rng_key, n_samples))

    return vmap(
        lambda samples, rng_key : posterior_predictive(
            model, rng_key, samples, stat_fun,
            *model_args, **model_kwargs)
    )(*vmap_args)


def print_summary(samples, divergences, prob=0.9):
    fields = ['beta_dyad', 'recip_coef', 'dist_coef', 
              's_var', 'r_var', 'sr_corr', 'z_var']
    samples = {k: v for k, v in samples.items() if 
        k in fields and k in samples.keys()}
    samples = jax.tree_map(lambda x : jnp.expand_dims(x, axis=0), samples)
    samples = OrderedDict({k: samples[k] for k in samples.keys()})

    numpyro_print_summary(samples, prob=prob)
    if divergences is not None:
        print(f"Number of divergences: {divergences}")


def pairwise_distance(U):
    U_norm_sq = jnp.sum(U ** 2, axis=1).reshape(-1, 1)
    dist_sq = U_norm_sq + U_norm_sq.T - 2 * U @ U.T
    return dist_sq


def initialize_parameters(Y, X_dyad=None, n_features=2, random_state=None):
    n_nodes = Y.shape[0]

    # MDS estimates for latent positions
    X_init = initialize_mds(
        Y, n_features=n_features, random_state=random_state)
    
    # bayes proportion (alpha = 1, beta = 1)
    s_init = logit((Y.sum(axis=1) + 1) / (n_nodes + 2))
    r_init = logit((Y.sum(axis=0) + 1) / (n_nodes + 2))
    sr_init = np.hstack((s_init.reshape(-1, 1), r_init.reshape(-1, 1)))
    edge_init = np.mean(sr_init)
    sr_init -= edge_init
    
    # logreg estimate of odds ratio
    dist = np.sqrt(
        adjacency_to_dyads(pairwise_distance(X_init), n=Y.shape[0])[:, 0])
    dyads = adjacency_to_dyads(Y, n=Y.shape[0])
    features = np.hstack((dyads[:, 0].reshape(-1, 1),
        (dyads[:, 0] * dist).reshape(-1, 1)))

    coefs = LogisticRegression(penalty=None).fit(
            features, dyads[:, 1]).coef_.ravel()
    recip_init, dist_init = coefs[0], coefs[1]

    # logreg estimate of coviariate effects
    if X_dyad is not None:
        nondiag = nondiag_indices(n_nodes)
        y = Y[nondiag]
        X_logreg = np.zeros((y.shape[0], X_dyad.shape[-1]))
        for k in range(X_dyad.shape[-1]):
            X_logreg[:, k] = X_dyad[..., k][nondiag]
        
        beta_dyad = LogisticRegression(penalty=None).fit(
                X_logreg, y).coef_.ravel()
    else:
        beta_dyad = 0.
             
    return {"U": X_init, 'z_sr': sr_init.T, 
            'mu_sr': edge_init, 'recip_coef': recip_init,
            'dist_coef': dist_init, 'sigma_sr': np.var(sr_init, axis=0),
            'rho_sr': np.corrcoef(s_init, r_init)[1, 0],
            'z_var': np.var(X_init), 'beta_dyad': beta_dyad}


    
def rlsm(Y, X_dyad, n_nodes, n_features=2, 
         reciprocity_type='distance', is_predictive=False):
    
    # cholesky decomposition of sender/receiver covariance
    rho_sr = numpyro.sample("rho_sr", dist.Uniform(-1., 1.))
    sigma_sr = numpyro.sample("sigma_sr",
        dist.InverseGamma(1.5 * jnp.ones(2), 1.5 * jnp.ones(2)))

    # centered sender/receiver parameters
    L_sr = jnp.eye(2).at[1,0].set(rho_sr)
    L_sr = jnp.diag(jnp.sqrt(sigma_sr)) @ L_sr
    Sigma_sr = L_sr @ L_sr.T
    numpyro.deterministic('s_var', Sigma_sr[0, 0])
    numpyro.deterministic('r_var', Sigma_sr[1, 1])
    numpyro.deterministic('sr_cov', Sigma_sr[0, 1])
    numpyro.deterministic('sr_corr', rho_sr)
    z_sr = numpyro.sample("z_sr",
        dist.Normal(jnp.zeros((2, n_nodes)), jnp.ones((2, n_nodes))))
    
    # soft intercept
    mu_sr = numpyro.sample('mu_sr', dist.Normal(0, 10.))

    # generate sender/receiver effects
    sr = mu_sr + L_sr @ z_sr
    s = numpyro.deterministic("s", sr[0]).reshape(-1, 1)
    r = numpyro.deterministic("r", sr[1]).reshape(-1, 1)

    # latent positions
    #u_sigma = numpyro.sample('u_sigma', dist.Gamma(1., 0.5))
    z_sigma = numpyro.sample('z_var', dist.InverseGamma(1.5, 1.5)) # 1 * invX2(1)
    if n_features is not None and n_features > 0 :
        U = numpyro.sample('U',
            dist.Normal(jnp.zeros((n_nodes, n_features)),
                        jnp.ones((n_nodes, n_features))))
        Z = numpyro.deterministic('Z', jnp.sqrt(z_sigma) * U)
 
    # reciprocity
    if reciprocity_type in ['distance', 'constant']:
        recip_coef = numpyro.sample("recip_coef", dist.Normal(0., 10.))  
        if reciprocity_type == 'distance':
            dist_coef = numpyro.sample('dist_coef', dist.Normal(0., 10.))
        else:
            dist_coef = 0.

    distances_sq = pairwise_distance(Z)
    sr = s + r.T

    if X_dyad is not None:
        n_covariates = X_dyad.shape[-1]
        beta_dyad = numpyro.sample("beta_dyad",
            dist.Normal(jnp.zeros(n_covariates), 10 * jnp.ones(n_covariates)))

    if X_dyad is not None:
        sr += X_dyad @ beta_dyad

    # likelihood
    with numpyro.handlers.condition(data={"Y": Y}):
        if reciprocity_type in ['distance', 'constant']:
            sr = adjacency_to_dyads(sr, n_nodes)
            triu = jnp.triu_indices(n_nodes, k=1)
            distances = jnp.sqrt(distances_sq[triu])
            logits = to_logits(
                    recip_coef, sr, distances, dist_coef, reciprocity_type)
            y = numpyro.sample("Y", BivariateBernoulli(logits=logits))

            if is_predictive:
                probas = numpyro.deterministic('probas', 
                        to_probs(recip_coef, sr, 
                            distances, dist_coef, reciprocity_type))
        else:
            nondiag = nondiag_indices(n_nodes)
            distances = jnp.sqrt(distances_sq[nondiag])
            eta = sr[nondiag] - distances
            y = numpyro.sample("Y", dist.Bernoulli(logits=eta))
            if is_predictive:
                numpyro.deterministic("probas", expit(eta))


class ReciprocityLSM(object):
    def __init__(self,
                 n_features=2,
                 reciprocity_type='distance',
                 random_state=42):
        self.n_features = n_features
        self.reciprocity_type = reciprocity_type
        self.random_state = random_state

    @property
    def model_args_(self):
        n_nodes = self.samples_['Z'].shape[1]
        return (None, self.X_dyad_fit_, n_nodes, self.n_features, 
                self.reciprocity_type)

    @property
    def model_kwargs_(self):
        return {'is_predictive': True}

    def sample(self, Y, X_dyad=None, 
            n_warmup=1000, n_samples=1000, adapt_delta=0.8):

        n_nodes = Y.shape[0]
        self.Y_fit_ = Y.copy()

        if self.reciprocity_type in ['distance', 'constant']:
            y = adjacency_to_dyads_multinomial(Y, n_nodes)
        else:
            y = adjacency_to_vec(Y)
        
        self.X_dyad_fit_ = (
            jnp.asarray(X_dyad.copy()) if X_dyad is not None else None)
        
        # parameter initialization 
        init_params = initialize_parameters(
                Y, self.X_dyad_fit_, 
                n_features=self.n_features, random_state=self.random_state)

        # run mcmc sampler
        rng_key = random.PRNGKey(self.random_state)
        model_args = (
                y, self.X_dyad_fit_, 
                n_nodes, self.n_features, self.reciprocity_type, False)
         
        mcmc = MCMC(NUTS(rlsm, target_accept_prob=adapt_delta), 
                num_warmup=n_warmup, num_samples=n_samples, num_chains=1)
        mcmc.run(rng_key, *model_args, init_params=init_params)
        self.diverging_ = jnp.sum(mcmc.get_extra_fields()['diverging'])

        # extract/process samples
        self.samples_ = mcmc.get_samples()
        
        # calculate log density
        self.logp_ = vmap(
            lambda sample : log_density(
                rlsm, model_args=(
                    y, self.X_dyad_fit_, 
                    n_nodes, self.n_features, self.reciprocity_type), 
                model_kwargs={'is_predictive': False},
                params=sample)[0])(self.samples_)
        self.map_idx_ = np.argmax(self.logp_)
        self.samples_ = jax.tree_map(lambda x : np.array(x), self.samples_)
        
        Z_ref = self.samples_['Z'][self.map_idx_]
        for idx in range(self.samples_['Z'].shape[0]):
            R, _ = orthogonal_procrustes(self.samples_['Z'][idx], Z_ref)
            self.samples_['Z'][idx] = self.samples_['Z'][idx] @ R
        
        self.s_ = self.samples_['s'].mean(axis=0)
        self.s_var_ = self.samples_['s_var'].mean(axis=0)
        self.r_ = self.samples_['r'].mean(axis=0)
        self.r_var_ = self.samples_['r_var'].mean(axis=0)
        self.sr_corr_ = self.samples_['sr_corr'].mean(axis=0)
        self.Z_ = self.samples_['Z'].mean(axis=0)

        if self.X_dyad_fit_ is not None:
            self.beta_dyad_ = self.samples_['beta_dyad'].mean(axis=0)

        if self.reciprocity_type in ['distance', 'constant']:
            self.recip_coef_ = self.samples_['recip_coef'].mean()
            if self.reciprocity_type == 'distance':
                self.dist_coef_ = self.samples_['dist_coef'].mean()
            else:
                self.dist_coef_ = 0.

        self.probas_ = self.predict_proba()
        self.auc_ = roc_auc_score(adjacency_to_vec(self.Y_fit_), self.probas_)
        
        # AIC
        n_params = (2 + self.n_features) * n_nodes 
        if self.reciprocity_type == 'distance':
            n_params += 2
        elif self.reciprocity_type == 'constant':
            n_params += 1
        if self.X_dyad_fit_ is not None:
            n_params += self.X_dyad_fit_.shape[-1] 
        
        if self.reciprocity_type in ['distance', 'constant']:
            n_dyads = 0.5 * n_nodes * (n_nodes - 1) 
        else:
            n_dyads = n_nodes * (n_nodes - 1) 

        loglik = self.loglikelihood()
        self.aic_ = -2 * loglik + 2 * n_params
        self.bic_ = -2 * loglik + np.log(n_dyads) * n_params 

        return self

    def posterior_predictive(self, stat_fun, random_state=42):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['Z'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))

        return np.asarray(vmap(
            lambda samples, rng_key : posterior_predictive(
                rlsm, rng_key, samples, stat_fun,
                *self.model_args_, **self.model_kwargs_)
        )(*vmap_args))
    

    def predict_proba(self, random_state=42):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['Z'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))

        return np.asarray(vmap(
            lambda samples, rng_key : predict_proba(
                rlsm, rng_key, samples,
                *self.model_args_, **self.model_kwargs_)
        )(*vmap_args).mean(axis=0))

    def print_summary(self, proba=0.95):
        print(f"AUC: {self.auc_:.3f}, " 
              f"AIC: {self.aic_:.3f}, "
              f"BIC: {self.bic_:.3f}, "
              f"DIC: {self.dic():.3f}")
        print_summary(self.samples_, self.diverging_, prob=proba)

    def plot(self, Y_obs=None, **fig_kwargs):
        Y_obs = Y_obs if Y_obs is not None else self.Y_fit_
        return plot_model(self, Y_obs, **fig_kwargs)

    def loglikelihood(self):
        n_nodes = self.Y_fit_.shape[0]

        sr = self.s_.reshape(-1, 1) + self.r_.reshape(-1, 1).T
        if self.X_dyad_fit_ is not None:
            sr += self.X_dyad_fit_ @ self.beta_dyad_
        distances_sq = pairwise_distance(self.Z_)
        
 
        if self.reciprocity_type in ['distance', 'constant']:
            sr = adjacency_to_dyads(sr, n_nodes)
            triu = jnp.triu_indices(n_nodes, k=1)
            distances = jnp.sqrt(distances_sq[triu])
            logits = to_logits(
                self.recip_coef_, sr, distances, self.dist_coef_, 
                self.reciprocity_type)
            
            y = adjacency_to_dyads_multinomial(self.Y_fit_, n_nodes)
            dis = BivariateBernoulli(logits=logits)
        else:
            nondiag = nondiag_indices(n_nodes)
            logits = sr[nondiag] - jnp.sqrt(distances_sq[nondiag])
            y = adjacency_to_vec(self.Y_fit_)
            dis = dist.Bernoulli(logits=logits)
        
        loglik = dis.log_prob(y).sum()

        return np.asarray(loglik).item()
     
    def dic(self, random_state=0):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['Z'].shape[0]
        n_nodes = self.samples_['Z'].shape[1]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))
        
        if self.reciprocity_type in ['distance', 'constant']:
            y = adjacency_to_dyads_multinomial(self.Y_fit_, n_nodes)
        else:
            y = adjacency_to_vec(self.Y_fit_)

        model_args = (y, self.X_dyad_fit_, n_nodes, self.n_features, 
                self.reciprocity_type)
        
        loglik = vmap(
            lambda samples, rng_key : log_likelihood(
                rlsm, rng_key, samples,
                *model_args, **self.model_kwargs_))(*vmap_args)

        loglik_hat = self.loglikelihood()
        p_dic = 2 * (loglik_hat - loglik.sum(axis=1).mean()).item()

        return -2 * loglik_hat + 2 * p_dic 
