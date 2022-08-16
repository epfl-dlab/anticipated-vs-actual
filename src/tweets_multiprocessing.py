import os
from multiprocessing import Pool
import time

from constants import lengths_dataset_path

def get_files():
    return [os.path.join(lengths_dataset_path, f"lengths_derived_dataset_batch_{i+1}.parquet") for i in range(40)]

def multiprocess_batches(func, files, n_cores=10):
    p = Pool(n_cores)
    print('Parallelized on number of cores:', n_cores)
    start = time.time()
    output = p.map(func, files)
    p.close()
    p.join()
    end = time.time()
    elapsed = end - start
    print('Elapsed time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))
    return output
