import pandas as pd
import numpy as np
import glob 

from os.path import join


for n_nodes in [50, 100, 150, 200, 250]:
    data = []
    for res_file_name in glob.glob(join('output', f'n{n_nodes}', 'result_*csv')):
        data.append(pd.read_csv(res_file_name))
    data = pd.concat(data)
    data.to_csv(f'rlsm_data_{n_nodes}nodes.csv', index=False)
