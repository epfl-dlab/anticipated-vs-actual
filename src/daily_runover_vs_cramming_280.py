import argparse
import os
from functools import reduce
import warnings
import pandas as pd
from pandarallel import pandarallel
from scipy.optimize import OptimizeWarning

from measure import measure
from tweets_multiprocessing import get_files, multiprocess_batches
from daily_cramming import process_batch_daily

from constants import allowed_languages

warnings.filterwarnings('ignore', category=OptimizeWarning)

def before_daily_wrapper(x):
    return process_batch_daily(x, before_switch=True, langs=allowed_languages, sources="allowed")


def after_daily_wrapper(x):
    return process_batch_daily(x, before_switch=False, langs=allowed_languages, sources="allowed")


def runover_280_wrapper(day_hist):
    return measure(day_hist=day_hist, measurement_at=280, measure='runover')


def both_280_wrapper(day_hist):
    return measure(day_hist=day_hist, measurement_at=280, measure='both')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Daily runover and cramming for languages that experienced the switch, sources: web and mobile together ')

    parser.add_argument('--n-cores', type=int, help='How many cores to use with multiprocessing, default: 20', default=20)
    parser.add_argument('--save-path', help='Where to save the output csv, default: ../data/measurements',
                        default='../data/measurements')

    args = parser.parse_args()
    pandarallel.initialize(nb_workers=args.n_cores)
    files = get_files()

    print("Calculating daily runover at 280 before the switch...")
    output_before = multiprocess_batches(before_daily_wrapper, files, n_cores=args.n_cores)
    before = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_before))
    before280_runover = before.parallel_apply(runover_280_wrapper, axis=1)

    before280_runover = before280_runover.to_frame().rename({0: "measurement"}, axis=1)
    before280_runover['measure'] = 'runover'

    print("Calculating daily runover and cramming at 280 after the switch...")
    output_after = multiprocess_batches(after_daily_wrapper, files, n_cores=args.n_cores)
    after = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_after))
    after280_both = after.parallel_apply(both_280_wrapper, axis=1)

    after280_both = pd.DataFrame(after280_both.tolist(),\
                             index=after280_both.index).rename({0: "runover", 1:"cramming"}, axis=1)

    df1 = after280_both.runover.to_frame().rename({'runover': 'measurement'}, axis=1)
    df2 = after280_both.cramming.to_frame().rename({'cramming': 'measurement'}, axis=1)

    df1['measure'] = 'runover'
    df2['measure'] = 'cramming'

    df = pd.concat([before280_runover, df1, df2])
    df.index = pd.to_datetime(df.index)
    df.to_csv(os.path.join(args.save_path, "daily_runover_vs_cramming.csv.gz"))
