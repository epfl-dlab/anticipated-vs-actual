import argparse
import os
import warnings
from functools import reduce

import numpy as np
import pandas as pd
from daily_runover_vs_cramming_280_per_source import before_daily_wrapper, after_daily_wrapper
from measure import measure
from pandarallel import pandarallel
from scipy.optimize import OptimizeWarning
from tweets_multiprocessing import get_files, multiprocess_batches

warnings.filterwarnings('ignore', category=OptimizeWarning)


def get_empirical_density(day_hist, x=140, limit=280):
    # fill missing values
    day_hist = day_hist.hist_chars.sort_index()[:limit] \
        .reindex(index=range(1, limit + 1), fill_value=0).fillna(0)
    density = (day_hist / np.sum(day_hist))
    return density[x:].sum()


def before140_both_wrapper(day_hist):
    return measure(day_hist=day_hist, measure='both', measurement_at=140)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Daily runover and cramming at 140 before the switch vs empirical fraction of tweets longer than 140'
                    ' after the switch for languages that experienced the switch, sources: web and mobile ')

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
    before = before.reset_index(level=0)
    before140_both = before.groupby("source_type").apply(lambda x: x.parallel_apply(before140_both_wrapper, axis=1)).T

    print("Calculating daily fraction of tweets longer than 140 after the switch...")
    output_after = multiprocess_batches(after_daily_wrapper, files, n_cores=args.n_cores)
    after = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_after))
    after = after.reset_index(level=0)
    after140_empirical = after.groupby("source_type").apply(lambda x: x.parallel_apply(get_empirical_density, axis=1)).T
    after140_empirical['measure'] = 'empirical fraction'

    web_before = pd.DataFrame(before140_both.web.tolist(), index=before140_both.index).rename(
        {0: "runover", 1: "cramming"}, axis=1)
    mobile_before = pd.DataFrame(before140_both.mobile.tolist(), index=before140_both.index).rename(
        {0: "runover", 1: "cramming"}, axis=1)

    df1 = web_before.runover
    df1.name = 'web'
    df2 = mobile_before.runover
    df2.name = 'mobile'

    runover_by_source = pd.concat((df2, df1), axis=1)
    runover_by_source['measure'] = 'runover'

    df1 = web_before.cramming
    df1.name = 'web'
    df2 = mobile_before.cramming
    df2.name = 'mobile'

    cramming_by_source = pd.concat((df2, df1), axis=1)
    cramming_by_source['measure'] = 'cramming'

    df = pd.concat([runover_by_source, cramming_by_source, after140_empirical])
    df.head()

    df = df.melt(id_vars=['measure'], \
                 ignore_index=False, value_name='measurement', var_name='source_type')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print("Saving to file...")
    df.to_csv(os.path.join(args.save_path, "daily_runover_vs_fraction_per_source.csv.gz"))
