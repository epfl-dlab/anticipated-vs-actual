import argparse
import os
import warnings
from functools import reduce

import numpy as np
import pandas as pd
from measure import measure
from pandarallel import pandarallel
from scipy.optimize import OptimizeWarning
from tweets_multiprocessing import get_files, multiprocess_batches
from daily_cramming import process_batch_daily
from constants import allowed_languages


warnings.filterwarnings('ignore', category=OptimizeWarning)


def get_empirical_density(day_hist, x=140, limit=280):
    # fill missing values
    day_hist = day_hist.hist_chars.sort_index()[:limit] \
        .reindex(index=range(1, limit + 1), fill_value=0).fillna(0)
    density = (day_hist / np.sum(day_hist))
    return density[x:].sum()


def before_daily_wrapper(x):
    return process_batch_daily(x, before_switch=True, langs=allowed_languages, sources="allowed")


def after_daily_wrapper(x):
    return process_batch_daily(x, before_switch=False, langs=allowed_languages, sources="allowed")


def runover_280_wrapper(day_hist):
    return measure(day_hist=day_hist, measurement_at=280, measure='runover')


def both_280_wrapper(day_hist):
    return measure(day_hist=day_hist, measurement_at=280, measure='both')


def before140_both_wrapper(day_hist):
    return measure(day_hist=day_hist, measure='both', measurement_at=140)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Daily runover and cramming at 140 before the switch vs empirical fraction'
                                                 ' of tweets longer than 140 after the switch for languages that experienced'
                                                 ' the switch,sources: web and mobile together')

    parser.add_argument('--n-cores', type=int, help='How many cores to use with multiprocessing, default: 20',
                        default=20)
    parser.add_argument('--save-path', help='Where to save the output csv, default: ../data/measurements',
                        default='../data/measurements')


    args = parser.parse_args()
    pandarallel.initialize(nb_workers=args.n_cores)
    files = get_files()

    print("Calculating daily runover and cramming at 140 before the switch...")
    output_before = multiprocess_batches(before_daily_wrapper, files, n_cores=args.n_cores)
    before = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_before))
    before140_both = before.parallel_apply(before140_both_wrapper, axis=1)
    before140_both = pd.DataFrame(before140_both.tolist(),\
                                index=before140_both.index).rename({0: "runover", 1:"cramming"}, axis=1)
    df1 = before140_both.runover.to_frame().rename({'runover': 'measurement'}, axis=1)
    df2 = before140_both.cramming.to_frame().rename({'cramming': 'measurement'}, axis=1)
    df1['measure'] = 'runover'
    df2['measure'] = 'cramming'

    print("Calculating daily fraction of tweets longer than 140 after the switch...")
    output_after = multiprocess_batches(after_daily_wrapper, files, n_cores=args.n_cores)
    after = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_after))
    after140_empirical = after.parallel_apply(get_empirical_density, axis=1)
    after140_empirical = after140_empirical.to_frame().rename({0: "measurement"}, axis=1)
    after140_empirical['measure'] = 'empirical fraction'

    df = pd.concat([df1, df2, after140_empirical])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print("Saving to file...")
    df.to_csv(os.path.join(args.save_path, "daily_runover_vs_fraction.csv.gz"))
