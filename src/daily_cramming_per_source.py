import argparse
import os
from functools import reduce
import warnings
import pandas as pd
from pandarallel import pandarallel
from scipy.optimize import OptimizeWarning

from measure import cramming_140_wrapper, cramming_280_wrapper
from tweets_multiprocessing import get_files, multiprocess_batches
from daily_cramming import process_batch_daily

from constants import allowed_languages

warnings.filterwarnings('ignore', category=OptimizeWarning)

def before_daily_wrapper(x):
    return process_batch_daily(x, before_switch=True, langs=allowed_languages, sources="all", groupby_cols='source')


def after_daily_wrapper(x):
    return process_batch_daily(x, before_switch=False, langs=allowed_languages, sources="all", groupby_cols='source')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Daily cramming per source: web, mobile and automated for languages that experienced the switch')

    parser.add_argument('--n-cores', type=int, help='How many cores to use with multiprocessing, default: 20', default=20)
    parser.add_argument('--save-path', help='Where to save the output csv, default: ../data/measurements',
                        default='../data/measurements')

    args = parser.parse_args()
    pandarallel.initialize(nb_workers=args.n_cores)
    files = get_files()

    print("Calculating daily cramming at 140 before the switch per source...")
    output_before = multiprocess_batches(before_daily_wrapper, files, n_cores=args.n_cores)
    before = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_before))
    before = before.reset_index(level=0)
    before = before.groupby("source_type")\
            .apply(lambda x: x.parallel_apply(cramming_140_wrapper, axis=1)).T

    print("Calculating daily cramming at 140 and 280 after the switch per source...")
    output_after = multiprocess_batches(after_daily_wrapper, files, n_cores=args.n_cores)
    after = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_after))
    after = after.reset_index(level=0)
    after_280 = after.groupby("source_type")\
            .apply(lambda x: x.parallel_apply(cramming_280_wrapper, axis=1)).T
    after_140 = after.groupby("source_type")\
            .apply(lambda x: x.parallel_apply(cramming_140_wrapper, axis=1)).T

    before['cramming_at'] = 140
    after_140['cramming_at'] = 140
    after_280['cramming_at'] = 280

    df = pd.concat([before, after_140, after_280])
    df = df.melt(id_vars=['cramming_at'],\
                ignore_index=False, value_name='cramming')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print("Saving to file...")
    df.to_csv(os.path.join(args.save_path, "fig9_daily_cramming_per_source.csv.gz"))
