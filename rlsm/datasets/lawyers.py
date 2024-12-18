import numpy as np
import pandas as pd

from os.path import dirname, join


def load_lawyers():
    """Lazegas lawyers friendship network."""
    module_path = dirname(__file__)
    dir_path = join(module_path, 'data', 'lazegas_lawyers')
    Y = np.loadtxt(join(dir_path, 'Eladv.dat'))
    features = pd.read_csv(join(dir_path, 'covariates.csv'))
    
    return Y, features
