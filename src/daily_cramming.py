import argparse
import os
from functools import reduce

import pandas as pd
from measure import cramming_140_wrapper, cramming_280_wrapper
from pandarallel import pandarallel
from tweets_multiprocessing import get_files, multiprocess_batches

from constants import allowed_languages, mobile_sources, web_sources


def before_daily_wrapper(x):
    return process_batch_daily(x, langs=allowed_languages, before_switch=True)


def after_daily_wrapper(x):
    return process_batch_daily(x, langs=allowed_languages, before_switch=False)


def process_batch_daily(file, before_switch, langs, sources=None, groupby_cols='date'):
    df = pd.read_parquet(file, columns=['lang', 'n_chars', "source", "created_at"])
    df = df.loc[df.lang.isin(langs)]
    if sources == 'allowed':
        df['source_type'] = df['source'].apply(convert_source)  
        df = df.loc[df['source_type'].isin(['web', 'mobile'])]
    elif sources == 'all':
        df['source_type'] = df['source'].apply(convert_source)  
    if before_switch:
        df = df[df["created_at"] < "2017-11-07"]
    else:
        df = df[df["created_at"] >= "2017-11-07"]
    groupby_obj = df.groupby(df.created_at.dt.date)
    if groupby_cols == "lang":
        groupby_obj = df.groupby([df.lang, df.created_at.dt.date])
    elif groupby_cols == "source":
        groupby_obj = df.groupby([df.source_type, df.created_at.dt.date])
    elif groupby_cols == ["source", "lang"]:
        groupby_obj = df.groupby([df.source_type, df.lang, df.created_at.dt.date])
    return groupby_obj.apply(lambda x: pd.Series({
        'hist_chars': x['n_chars'].value_counts()
    }))

def convert_source(text):
    if text in mobile_sources:
        return 'mobile'
    elif text in web_sources:
        return 'web'
    else:
        return 'automated'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Daily cramming for languages that experienced the switch and all sources together: web, mobile and automated ')

    parser.add_argument('--n-cores', type=int, help='How many cores to use with multiprocessing, default: 20', default=20)
    parser.add_argument('--save-path', help='Where to save the output csv, default: ../data/measurements',
                        default='../data/measurements')

    args = parser.parse_args()
    pandarallel.initialize(nb_workers=args.n_cores)
    files = get_files()

    print("Calculating daily cramming at 140 before the switch...")
    output_before = multiprocess_batches(before_daily_wrapper, files, n_cores=args.n_cores)
    before = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_before))
    before = before.parallel_apply(cramming_140_wrapper, axis=1).to_frame().rename({0: "cramming"}, axis=1)

    print("Calculating daily cramming at 140 and 280 after the switch...")
    output_after = multiprocess_batches(after_daily_wrapper, files, n_cores=args.n_cores)
    after = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_after))
    after_280 = after.parallel_apply(cramming_280_wrapper, axis=1).to_frame().rename({0: "cramming"}, axis=1)
    after_140 = after.parallel_apply(cramming_140_wrapper, axis=1).to_frame().rename({0: "cramming"}, axis=1)

    after_140['cramming_at'] = 140
    before['cramming_at'] = 140
    after_280['cramming_at'] = 280

    df = pd.concat([before, after_140, after_280])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print("Saving to file...")
    df.to_csv(os.path.join(args.save_path, "fig3_fig4_daily_cramming_allowed_langs_all_source.csv.gz"))
