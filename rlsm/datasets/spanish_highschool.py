import numpy as np
import pandas as pd
import networkx as nx

from os.path import dirname, join


def load_spanish_highschool():
    """Spanish highschool relationship network."""
    module_path = dirname(__file__)
    dir_path = join(module_path, 'data', 'spanish_highschool_id2')
    
    data = pd.read_csv(join(dir_path, 'edges.csv'))
    edgelist = data[['source', 'target', 'weight']]
    g = nx.from_pandas_edgelist(
            edgelist, edge_attr='weight', create_using=nx.DiGraph)
    Y = nx.to_numpy_array(g)

    features = pd.read_csv(join(dir_path, 'nodes.csv'))
    features = features[['course', 'cluster', 'sex']]  
    return Y, features
