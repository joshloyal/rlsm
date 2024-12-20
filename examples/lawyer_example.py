import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

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

# run the MCMC algorithm for 2,500 warmup iterations and collect 2,500 post warmup samples
lsm.sample(Y, X_dyad=X_dyad, n_warmup=2500, n_samples=2500)

# summary of the posterior distribution
lsm.print_summary()

# diagnostic plots
lsm.plot(figsize=(12, 9))
plt.savefig("lawyer_lsm.png", dpi=300, bbox_inches='tight')

# plot the observed network using the inferred latent positions
fig, ax = plt.subplots(figsize=(6, 6))

# create a networkx graph
g = nx.from_numpy_array(Y, create_using=nx.DiGraph)

# the posterior mean of the latent positions are stored in lsm.Z_
pos = {k : lsm.Z_[k] for k in range(Y.shape[0])}

# color code edges based on whether they are mutual or not
ecolor = ['darkorange' if Y[e[1], e[0]] else 'k' for e in list(nx.to_edgelist(g))]

# plot network embedding
colors = np.asarray(["tomato", "steelblue", "goldenrod"])
nx.draw_networkx(g, pos, 
                 edgecolors='k',
                 node_color=colors[features['office'].values - 1],
                 edge_color=ecolor, width=1.0, with_labels=False,
                 arrowsize=5, node_size=50, ax=ax)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=16)
ax.set_ylabel('Latent Dimension 2', fontsize=18)
ax.set_xlabel('Latent Dimension 1', fontsize=18)

fig.savefig('lawyer_embed.png', dpi=300, bbox_inches='tight')
