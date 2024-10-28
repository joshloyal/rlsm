import numpy as np
import pandas as pd
import networkx as nx

from os.path import dirname, join


def load_addhealth():
    module_path = dirname(__file__)
    dir_path = join(module_path, 'data', 'addhealth')

    data = pd.read_csv(join(dir_path, 'comm71.txt'), 
        delim_whitespace=True, skiprows=4, header=None, 
        names=['source', 'target', 'weight'])
    
    features = pd.read_csv(join(dir_path, 'comm71_att.txt'),
            delim_whitespace=True, skiprows=10, header=None, 
            names=['sex', 'race', 'grade'])

    g = nx.from_pandas_edgelist(data, create_using=nx.DiGraph)
    node_ids = np.asarray(sorted(g.nodes()))

    Y = nx.to_numpy_array(g, node_ids)
    features = features.iloc[node_ids-1]
    
    # only include non-isolated nodes
    nonisolated_nodes = np.where(Y.sum(axis=1) + Y.sum(axis=0) > 0)[0]
    Y = Y[nonisolated_nodes, :][:, nonisolated_nodes]
    features = features.iloc[nonisolated_nodes]

    return Y, features

