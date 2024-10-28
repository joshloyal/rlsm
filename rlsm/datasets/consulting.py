import numpy as np
import pandas as pd
import networkx as nx

from os.path import dirname, join


def load_consulting():
    """Spanish highschool relationship network."""
    module_path = dirname(__file__)
    dir_path = join(module_path, 'data')
    
    data = pd.read_table(join(dir_path, 'consulting3.txt'), 
            header=None, sep=' ').loc[:, 1:]
    data.columns = ['a', 'b', 'weight']
    g = nx.from_pandas_edgelist(data, source='a', target='b', 
            edge_attr='weight', create_using=nx.DiGraph)


    Y = nx.to_numpy_array(g, sorted(g.nodes()), weight='weight')
    #Y = (Y >= 3).astype(int)
    Y = (Y > 3).astype(int)
    
    locations = np.loadtxt(join(dir_path, 'consulting-location.txt'))

    return Y, locations
