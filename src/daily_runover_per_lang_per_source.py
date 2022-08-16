import argparse
import os
from functools import reduce
import warnings
import pandas as pd
from pandarallel import pandarallel
from scipy.optimize import OptimizeWarning

from daily_runover_vs_cramming_280 import runover_280_wrapper
from tweets_multiprocessing import get_files, multiprocess_batches
from daily_cramming import process_batch_daily

from constants import lang_sorted

warnings.filterwarnings('ignore', category=OptimizeWarning)


def after_daily_wrapper(x):
    return process_batch_daily(x, before_switch=False, langs=lang_sorted,\
                               sources="allowed", groupby_cols=["source", "lang"])


def postprocess(df, after, source):
    new_index = pd.date_range(pd.Timestamp('2017-01-01 00:00:00'), pd.Timestamp(after.index.max()), freq='D')
    df = df.reindex(new_index).reset_index().rename({"index": "date"}, axis=1).reset_index().set_index("date")
    if source == 'web':
        df.columns = df.columns.droplevel(0)
        df = df.rename_axis(None, axis=1)
        df = df.rename({'': "index"}, axis=1)
    df = df.melt(id_vars=['index'], value_vars=['ar', 'de', 'en', 'es', 'et', 'fa', 'fr', 'hi',\
                                                'ht', 'in', 'it', 'ja', 'ko', 'nl', 'pl', 'pt',\
                                                'ru', 'sv', 'th', 'tl', 'tr', 'ur', 'zh'],\
                        ignore_index=False, value_name='runover280', var_name='lang')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.loc[df.index < "2017-11-07"] = df.loc[df.index < "2017-11-07"].fillna(0)
    return df

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Daily runover for all languages, sources: web and mobile separately')

    parser.add_argument('--n-cores', type=int, help='How many cores to use with multiprocessing, default: 20', default=20)
    parser.add_argument('--save-path', help='Where to save the output csv, default: ../data/measurements',
                        default='../data/measurements')

    args = parser.parse_args()
    pandarallel.initialize(nb_workers=args.n_cores)
    files = get_files()

    print("Calculating daily runover at 280 per language...")
    output_after = multiprocess_batches(after_daily_wrapper, files, n_cores=args.n_cores)
    after = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_after))
    after = after.reset_index(level=[0, 1])

    web = after[after.source_type == 'web'].drop("source_type", axis=1)
    mobile = after[after.source_type == 'mobile'].drop("source_type", axis=1)
    web = web.groupby("lang").apply(lambda x: x.parallel_apply(runover_280_wrapper, axis=1))
    mobile = mobile.groupby("lang").apply(lambda x: x.parallel_apply(runover_280_wrapper, axis=1)).T
    web = web.reset_index(level=0).pivot(columns='lang')

    df_web = postprocess(web, after, source='web')
    df_mobile = postprocess(mobile, after, source='mobile')
    df_web['source'] = 'web'
    df_mobile['source'] = 'mobile'
    df = pd.concat([df_web, df_mobile])

    print("Saving to file...")
    df.to_csv(os.path.join(args.save_path, "daily_runover_per_lang_per_source.csv.gz"))
