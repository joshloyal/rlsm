import numpy as np
import pandas as pd
import os

from scipy.linalg import orthogonal_procrustes
from scipy.stats import pearsonr

from rlsm import ReciprocityLSM
from rlsm.datasets import generate_data


def simulation(seed=42, n_nodes=250):
    Y, X_dyad, params = generate_data(n_nodes=n_nodes, density=0.2, odds_ratio=2, mu=1, include_covariates=True, random_state=seed)
    
    model = ReciprocityLSM(n_features=2, reciprocity_type='distance', random_state=42)
    model.sample(Y, X_dyad=X_dyad, n_warmup=2500, n_samples=5000)

    recip_coef = params['recip_coef']
    dist_coef = params['dist_coef']
    s_var = params['s_var']
    r_var = params['r_var']
    sr_corr = params['sr_corr']


    s_var_est = np.mean(model.samples_['s_var'])
    r_var_est = np.mean(model.samples_['r_var'])
    sr_corr_est = np.mean(model.samples_['sr_corr'])


    recip_coef_mse = np.abs(recip_coef - model.recip_coef_)
    dist_coef_mse = np.abs(dist_coef - model.dist_coef_)
    s_var_mse = np.abs(s_var - s_var_est)
    r_var_mse = np.abs(r_var - r_var_est)
    sr_corr_mse = np.abs(sr_corr - sr_corr_est)
    sd_ef_mse = np.mean((model.s_ - params['s'].ravel()) ** 2)
    rc_ef_mse = np.mean((model.r_ - params['r'].ravel()) ** 2)
    sd_ef_pc = np.corrcoef(model.s_, params['s'].ravel())[0,1]
    rc_ef_pc = np.corrcoef(model.r_, params['r'].ravel())[0,1]
    beta_mse = np.mean((model.beta_dyad_ - params['beta_dyad']) ** 2)

    R, _ = orthogonal_procrustes(model.Z_, params['Z'])
    Z_est = model.Z_ @ R
    Z_mse = np.mean((Z_est - params['Z']) ** 2)
    
    pearson_corr, _ = pearsonr(model.predict_proba(), params['probas'])

    data = {
            'n_nodes': n_nodes, 
            'seed': seed, 
            'Pearson correlation coefficient': pearson_corr, 
            'recip_coef_mse': recip_coef_mse.item(), 
            'dist_coef_mse': dist_coef_mse.item(), 
            's_var_mse': s_var_mse.item(), 
            'r_var_mse': r_var_mse.item(), 
            'sr_corr_mse': sr_corr_mse.item(), 
            'Z_mse': Z_mse.item(), 
            'Sender-effect mse': sd_ef_mse.item(), 
            'Receiver-effect mse': rc_ef_mse.item(), 
            'Sender-effect PC': sd_ef_pc.item(), 
            'Receiver-effect PC': rc_ef_pc.item(), 
            'beta_mse': beta_mse.item()
    }

    
    data = pd.DataFrame(data, index=[0])

    dir_base = 'output'
    out_file = f'result_{seed}.csv'
    dir_name = os.path.join(dir_base, f"n{n_nodes}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
for i in range(50):
    for n_nodes in [50, 100, 150, 200, 250]:
        simulation(seed=i, n_nodes=n_nodes)
