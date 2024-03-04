import matplotlib.pyplot as plt
import numpy as np
import arviz as az

from .gof import (
    std_row_mean,
    std_col_mean,
    reciprocity,
    cycle_dependence,
    trans_dependence
)
from .network_utils import adjacency_to_vec


def plot_model(rlsm, Y_obs, **fig_kwargs):
    if rlsm.reciprocity_type == 'none':
        ax_dict = plt.figure(
            constrained_layout=True, **fig_kwargs).subplot_mosaic(
            """
            AABC
            DDEE
            FFGH
            """
        )
    else:
        ax_dict = plt.figure(
            constrained_layout=True, **fig_kwargs).subplot_mosaic(
            """
            AABC
            DDEE
            FGHI
            """
        )
    
    ax = list(ax_dict.values())
    

    ax[0].plot(np.asarray(rlsm.logp_), alpha=0.8)
    ax[0].axhline(
        np.asarray(rlsm.logp_).mean(), color='k', linestyle='--')
    ax[0].set_ylabel('Log-Posterior')
    
    for param in ['s_var', 'r_var', 'sr_corr', 'z_sigma']:
        n_samples = rlsm.samples_[param].shape[0]
        ax[1].plot(np.asarray(rlsm.samples_[param]), alpha=0.8)
        param_mean = np.asarray(rlsm.samples_[param]).mean()
        ax[1].axhline(param_mean, color='k', linestyle='--')
        label = param.split('_')[0]
        #ax[1].text(x=n_samples, y=param_mean,
        #    s=r"$\sigma_{{{}}}$".format(label))

    ax[1].set_ylabel('Marginal Variances')
    
    ax[2].plot(np.asarray(rlsm.samples_['mu_sr']), alpha=0.8)
    ax[2].axhline(
        np.asarray(rlsm.samples_['mu_sr']).mean(), color='k', linestyle='--')
    if rlsm.reciprocity_type in ['distance', 'common']:
        ax[2].plot(np.asarray(rlsm.samples_['recip_coef']), alpha=0.8)
        ax[2].axhline(
            np.asarray(rlsm.samples_['recip_coef']).mean(), color='k', linestyle='--')
    if rlsm.reciprocity_type == 'distance':
        ax[2].plot(np.asarray(rlsm.samples_['dist_coef']), alpha=0.8)
        ax[2].axhline(
            np.asarray(rlsm.samples_['dist_coef']).mean(), color='k', linestyle='--')

    ax[2].set_ylabel('Coefficients')
    
    if rlsm.reciprocity_type in ['distance', 'common']:
        phi = rlsm.samples_['dist_coef']
        rho = rlsm.samples_['recip_coef']
        az.plot_kde(phi, rho, hdi_probs=[0.25, 0.75, 0.95],
                contourf_kwargs={"cmap": "Blues"}, ax=ax[3])
        ax[3].scatter(phi.mean(), rho.mean(), color='k', marker='x')
        ax[3].set_xlabel(r"Distance-Dependent Reciprocity ($\phi$)")
        ax[3].set_ylabel(r"Baseline Reciprocity ($\rho$)")

    y_vec = adjacency_to_vec(Y_obs)
    if rlsm.reciprocity_type == 'none':
        stats = {
            'sd.rowmean': std_row_mean,
            'sd.colmean': std_col_mean,
            'reciprocity': reciprocity,
            'cycles': cycle_dependence,
            'transitivity': trans_dependence
        }
        start = 3  
    else:
        stats = {
            'reciprocity': reciprocity,
            'sd.rowmean': std_row_mean,
            'sd.colmean': std_col_mean,
            'cycles': cycle_dependence,
            'transitivity': trans_dependence
        }
        start = 4 
    for k, (key, stat_func) in enumerate(stats.items()):
        res = rlsm.posterior_predictive(stat_func)
        ax[k+start].hist(res, edgecolor='k', color='#add8e6')
        ax[k+start].axvline(
            stat_func(y_vec), color='k', linestyle='--', linewidth=3)
        ax[k+start].set_xlabel(key)

    return ax_dict
