import numpy as np
import pandas as pd

from os.path import dirname, join


def load_lawyers(relation_type='advise'):
    """Lazegas lawyers friendship network."""
    module_path = dirname(__file__)
    dir_path = join(module_path, 'data', 'lazegas_lawyers')

    if relation_type == 'advise':
        Y = np.loadtxt(join(dir_path, 'adj_2.npy'))
    else:
        Y = np.loadtxt(join(dir_path, 'adj_3.npy'))

    features = pd.read_csv(join(dir_path, 'covariates.csv'))
    
    return Y, features
