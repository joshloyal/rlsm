import jax.numpy as jnp
import jax

from .network_utils import adjacency_to_vec, vec_to_adjacency, multinomial_to_adjacency


def nancor(a, b):
    ab = (a - jnp.nanmean(a)) * (b - jnp.nanmean(b))
    return jnp.nanmean(ab) / (jnp.nanstd(a) * jnp.nanstd(b))


def density(y_vec):
    return y_vec.mean()


def std_row_mean(y_vec, is_vec=False):
    Y = vec_to_adjacency(y_vec, include_nan=True)
    # ignore diagonal entries
    return jnp.nanstd(jnp.nanmean(Y, axis=1), ddof=1)


def std_col_mean(y_vec):
    Y = vec_to_adjacency(y_vec, include_nan=True)
    # ignore diagonal entries
    return jnp.nanstd(jnp.nanmean(Y, axis=0), ddof=1)


#def reciprocity(y_vec, is_adj=False):
#    """Correlation between vec(Y) and vec(Y.T)"""
#    Y = y_vec if is_adj else vec_to_adjacency(y_vec, include_nan=True)
#    return nancor(Y.T.ravel(), Y.ravel())


def reciprocity(y_vec, is_adj=False):
    """Percentage of mutual ties"""
    Y = y_vec if is_adj else vec_to_adjacency(y_vec, include_nan=False)
    return jnp.sum(Y * Y.T) / jnp.sum(Y)


def cycle_dependence(y_vec):
    #"""i->j and j->k and k->i"""
    """#{i->j and j->k and k->i} / #{2-paths, i.e., possible transitive triples}"""
    Y = vec_to_adjacency(y_vec, include_nan=False)
    #E = Y - density(y_vec)
    #E = E.at[jnp.isnan(E)].set(0)
    #E = E.at[jnp.diag_indices(Y.shape[0])].set(0)
    #D = (~jnp.isnan(E)).astype(int)
    #num = jnp.trace(E @ E @ E)
    #den = jnp.trace(D @ D @ D) * jnp.nanstd(Y.T.ravel(), ddof=1) ** 3
    Y_sq = Y @ Y
    num = jnp.sum(Y_sq * Y.T)
    den = jnp.sum(Y_sq) - jnp.trace(Y_sq)
    return num / den


def trans_dependence(y_vec):
    """#{i->j and j->k and i->k} / #{2-paths, i.e., possible transitive triples}"""
    Y = vec_to_adjacency(y_vec, include_nan=False)
    #E = Y - density(y_vec)
    ##E = E.at[jnp.isnan(E)].set(0)
    #E = E.at[jnp.diag_indices(Y.shape[0])].set(0)
    #D = (~jnp.isnan(E)).astype(int)
    #num = jnp.trace(E @ E.T @ E)
    #den = jnp.trace(D @ D.T @ D) * jnp.nanstd(Y.T.ravel(), ddof=1) ** 3
    Y_sq = Y @ Y
    num = jnp.sum(Y_sq * Y)
    den = jnp.sum(Y_sq) - jnp.trace(Y_sq)
    return num / den
