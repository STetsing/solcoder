import numpy as np
import pandas as pd
from multiprocessing import  Pool, Manager
from functools import partial

def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data_split = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data_split

def run_on_subset(func, data_subset):
    data_subset['code_and_comment'] = data_subset.apply(func, axis=1)
    print(f'Job {os.getpid()} finished')
    return data_subset


def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)
