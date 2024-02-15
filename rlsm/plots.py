import matplotlib.pyplot as plt
import numpy as np

from .gof import (
    std_row_mean,
    std_col_mean,
    reciprocity,
    cycle_dependence,
    trans_dependence
)
from .network_utils import adjacency_to_vec


def plot_model(ame, Y_obs, **fig_kwargs):
    ax_dict = plt.figure(
        constrained_layout=True, **fig_kwargs).subplot_mosaic(
        """
        AABC
        DDEE
        FFGH
        """
    )
    
    ax = list(ax_dict.values())
    

    ax[0].plot(np.asarray(ame.logp_), alpha=0.8)
    ax[0].axhline(
        np.asarray(ame.logp_).mean(), color='k', linestyle='--')
    ax[0].set_ylabel('Log-Posterior')
    
    for param in ['a_sigma', 'b_sigma', 'ab_sigma', 'u_sigma']:
        n_samples = ame.samples_[param].shape[0]
        ax[1].plot(np.asarray(ame.samples_[param]), alpha=0.8)
        param_mean = np.asarray(ame.samples_[param]).mean()
        ax[1].axhline(param_mean, color='k', linestyle='--')
        label = param.split('_')[0]
        ax[1].text(x=n_samples, y=param_mean,
            s=r"$\sigma_{{{}}}$".format(label))

    ax[1].set_ylabel('Marginal Variances')
    
    ax[2].plot(np.asarray(ame.samples_['recip_coef']), alpha=0.8)
    ax[2].axhline(
        np.asarray(ame.samples_['recip_coef']).mean(), color='k', linestyle='--')
    ax[2].plot(np.asarray(ame.samples_['dist_coef']), alpha=0.8)
    ax[2].axhline(
        np.asarray(ame.samples_['dist_coef']).mean(), color='k', linestyle='--') 
    ax[2].set_ylabel('Coefficients')
    
    y_vec = adjacency_to_vec(Y_obs)
    stats = {
        'sd.rowmean': std_row_mean,
        'sd.colmean': std_col_mean,
        'reciprocity': reciprocity,
        'cycles': cycle_dependence,
        'transitivity': trans_dependence
    }
    start = 3
    for k, (key, stat_func) in enumerate(stats.items()):
        res = ame.posterior_predictive(stat_func)
        ax[k+start].hist(res, edgecolor='k', color='#add8e6')
        ax[k+start].axvline(
            stat_func(y_vec), color='k', linestyle='--', linewidth=3)
        ax[k+start].set_xlabel(key)

    return ax_dict
