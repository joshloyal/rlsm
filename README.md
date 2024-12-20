[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/joshloyal/multidynet/blob/master/LICENSE)

## A Latent Space Approach to Inferring Distance-Dependent Reciprocity 

*Package Authors: [Joshua D. Loyal](https://joshloyal.github.io/) and Xiangyu Wu*

This package provides an interface for the model described in
"A Latent Space Approach to Inferring Distance-Dependent Reciprocity in Directed Networks." Inference is performed using
Hamiltonian Monte Carlo. For more details, see [Loyal et. al. (2024)](https://arxiv.org/abs/2411.18433).

Dependencies
------------
``rlsm`` requires:

- Python (>= 3.10)

and the requirements highlighted in [requirements.txt](requirements.txt). To install the requirements, run

```python
pip install -r requirements.txt
```

Installation
------------

You need a working installation of numpy, scipy, and jax to install ``rlsm``. Install these required dependencies before proceeding.  Use the following commands to get the copy from GitHub and install all the dependencies:

```
>>> git clone https://github.com/joshloyal/rlsm.git
>>> cd rlsm
>>> pip install -r requirements.txt
>>> python setup.py develop
```

Example
-------

```python
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
# NOTE: More warmup iterations and post-warmup samples should be collected!
lsm.sample(Y, X_dyad=X_dyad, n_warmup=1000, n_samples=1000)

# summary of the posterior distribution
lsm.print_summary()

#>>> AUC: 0.938, AIC: 2955.982, BIC: 4631.574, DIC: 2830.108
#>>> 
#>>>                   mean       std    median      2.5%     97.5%     n_eff     r_hat
#>>> beta_dyad[0]      0.51      0.15      0.51      0.21      0.79    832.92      1.00
#>>> beta_dyad[1]      1.79      0.13      1.79      1.55      2.08    795.70      1.00
#>>>    dist_coef     -0.10      0.19     -0.09     -0.48      0.26    321.51      1.00
#>>>        r_var      2.55      0.57      2.49      1.50      3.67    281.57      1.02
#>>>   recip_coef      0.96      0.45      0.98     -0.00      1.72    328.63      1.00
#>>>        s_var      2.27      0.58      2.19      1.32      3.49    212.63      1.00
#>>>      sr_corr     -0.51      0.18     -0.50     -0.89     -0.19    200.04      1.01
#>>>        z_var      4.65      0.88      4.59      2.97      6.26    338.39      1.02

# diagnostic plots
lsm.plot(figsize=(12, 9))
```

<img src="images/lawyer_lsm.png" width="100%" />


Simulation Studies and Real-Data Applications
---------------------------------------------

The [scripts](scripts) directory includes the simulation studies and real-data application found in the article.
