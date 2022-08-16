import argparse
import os
from functools import reduce
import warnings
import pandas as pd
from pandarallel import pandarallel
from scipy.optimize import OptimizeWarning

from measure import cramming_280_wrapper
from tweets_multiprocessing import get_files, multiprocess_batches
from daily_cramming import process_batch_daily

from constants import lang_sorted

warnings.filterwarnings('ignore', category=OptimizeWarning)


def after_daily_wrapper(x):
    return process_batch_daily(x, before_switch=False,  langs=lang_sorted, sources="allowed", groupby_cols='lang')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Daily cramming for all languages, sources: web and mobile together')

    parser.add_argument('--n-cores', type=int, help='How many cores to use with multiprocessing, default: 20', default=20)
    parser.add_argument('--save-path', help='Where to save the output csv, default: ../data/measurements',
                        default='../data/measurements')

    args = parser.parse_args()
    pandarallel.initialize(nb_workers=args.n_cores)
    files = get_files()

    print("Calculating daily cramming at 280 per language...")
    output_after = multiprocess_batches(after_daily_wrapper, files, n_cores=20)
    after = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_after))
    after = after.reset_index(level=0)
    after = after.groupby("lang").apply(lambda x: x.parallel_apply(cramming_280_wrapper, axis=1)).T

    new_index = pd.date_range(pd.Timestamp('2017-01-01 00:00:00'), pd.Timestamp(after.index.max()), freq='D')
    before_after = after.reindex(new_index).reset_index().rename({"index": "date"}, axis=1).reset_index().set_index("date")

    df = before_after.melt(id_vars=['index'], value_vars=['ar', 'de', 'en', 'es', 'et', 'fa', 'fr', 'hi',\
                                                          'ht', 'in', 'it', 'ja', 'ko', 'nl', 'pl', 'pt',\
                                                          'ru', 'sv', 'th', 'tl', 'tr', 'ur', 'zh'],\
                        ignore_index=False, value_name='cramming280')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.loc[df.index < "2017-11-07"] = df.loc[df.index < "2017-11-07"].fillna(0)

    print("Saving to file...")
    df.to_csv(os.path.join(args.save_path, "fig7_daily_cramming_all_langs.csv.gz"))
