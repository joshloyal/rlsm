import matplotlib.pyplot as plt
import numpy as np

from rlsm import ReciprocityLSM
from rlsm.datasets import load_lawyers
from rlsm.covariates import dyad_cat_diff, dyad_nominal_diff


# load the Lazega lawyers's advice network
Y, features = load_lawyers()

# binary directed adjacency matrix
Y.shape
#>>> (71, 71)

# create a matrix of edge-wise covariates 
X_dyad = np.dstack([
    dyad_cat_diff(features['female']),
    dyad_cat_diff(features['practice']),
])

X_dyad.shape
#>>> (71, 71, 2)

# initialize a distance-dependent LSM with d = 2 latent dimensions
# NOTE: reciprocity_type : str {'none', 'constant', 'distance'}
lsm = ReciprocityLSM(n_features=2, reciprocity_type='distance', random_state=42)

# run the MCMC algorithm for 1,000 warmup iterations and collect 1,000 post warmup samples
# NOTE: Typically more warmup iterations and post-warmup samples should be collected
lsm.sample(Y, X_dyad=X_dyad, n_warmup=1000, n_samples=1000)

# summary of the posterior distribution
lsm.print_summary()

# diagnostic plots
lsm.plot(figsize=(12, 9))
plt.savefig("lawyer_lsm.png", dpi=300, bbox_inches='tight')
