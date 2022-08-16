import os
from functools import reduce

import argparse
import pandas as pd
from measure import measure
from pandarallel import pandarallel
from tweets_multiprocessing import get_files, multiprocess_batches

from constants import allowed_sources, lang_sorted


def before_period_wrapper(x):
    return process_batch_period(x, before_switch=True)


def after_period_wrapper(x):
    return process_batch_period(x, before_switch=False)


def process_batch_period(file, before_switch):
    df = pd.read_parquet(file, columns=['lang', 'n_chars', "source", "created_at"])
    df = df.loc[df['lang'].isin(lang_sorted)]
    df = df.loc[df['source'].isin(allowed_sources)]
    if before_switch:
        df = df[(df["created_at"].dt.year == 2017) & (df["created_at"].dt.month < 11)]
    else:
        df = df[(df["created_at"].dt.year == 2019) & (df["created_at"].dt.month < 11)]
    return df.groupby("lang").apply(lambda x: x['n_chars'].value_counts().sort_index())


def lang_cramming_140_wrapper(lang_hist):
    day_hist = lang_hist.hist_chars.reset_index(level=0, drop=True).to_frame()
    return measure(day_hist=day_hist, measurement_at=140, measure='cramming')


def lang_cramming_280_wrapper(lang_hist):
    day_hist = lang_hist.hist_chars.reset_index(level=0, drop=True).to_frame()
    return measure(day_hist=day_hist, measurement_at=280, measure='cramming')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='General cramming for two control periods: January-October 2017 and January-October 2019'
                    ' per language, for languages that experienced the switch, sources: web and mobile')

    parser.add_argument('--n-cores', type=int, help='How many cores to use with multiprocessing, default: 20',
                        default=20)
    parser.add_argument('--save-path', help='Where to save the output csv, default: /scratch/czestoch/tweet-length',
                        default='/scratch/czestoch/tweet-length')

    args = parser.parse_args()
    pandarallel.initialize(nb_workers=args.n_cores)
    files = get_files()

    output_before = multiprocess_batches(before_period_wrapper, files, n_cores=args.n_cores)
    before = reduce(lambda a, b: pd.concat((a, b), axis=1).sum(axis=1), filter(lambda x: not x.empty, output_before))
    before = before.to_frame().rename({0: "hist_chars"}, axis=1)

    print("Fitting lognormal per language before the switch and estimate cramming at 140 per language...")
    cramming_before_140 = before.groupby("lang").parallel_apply(lang_cramming_140_wrapper)
    cramming_before_140.name = 'cramming_before_140'

    output_after = multiprocess_batches(after_period_wrapper, files, n_cores=args.n_cores)
    after = reduce(lambda a, b: pd.concat((a, b), axis=1).sum(axis=1), filter(lambda x: not x.empty, output_after))
    after = after.to_frame().rename({0: "hist_chars"}, axis=1)

    print("Fitting lognormal per language after the switch and estimate cramming at 280 per language...")
    cramming_after_280 = after.groupby("lang").parallel_apply(lang_cramming_280_wrapper)
    cramming_after_280.name = 'cramming_after_280'

    df = pd.merge(cramming_before_140.reset_index(), cramming_after_280.reset_index(), on='lang')

    print("Saving file...")
    df.to_csv(os.path.join(args.save_path, "fig6_cramming_per_lang_before_vs_after.csv.gz"), index=False)
