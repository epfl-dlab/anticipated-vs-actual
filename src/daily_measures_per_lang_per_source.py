import argparse
import os
from functools import reduce
import warnings
import pandas as pd
from pandarallel import pandarallel
from scipy.optimize import OptimizeWarning

from measure import cramming_140_wrapper, cramming_280_wrapper
from daily_runover_vs_cramming_280_per_source import runover_280_wrapper
from daily_runover_vs_fraction_140 import get_empirical_density
from tweets_multiprocessing import get_files, multiprocess_batches
from daily_cramming import process_batch_daily

from constants import lang_sorted

warnings.filterwarnings('ignore', category=OptimizeWarning)

def before_daily_wrapper(x):
    return process_batch_daily(x, before_switch=True, 
                               langs=lang_sorted, sources="all",
                               groupby_cols=["source", "lang"])


def after_daily_wrapper(x):
    return process_batch_daily(x, before_switch=False,
                               langs=lang_sorted, sources="all",
                               groupby_cols=["source", "lang"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Daily measures per source: web, mobile and automated and per language for all 23 languages')

    parser.add_argument('--n-cores', type=int, help='How many cores to use with multiprocessing, default: 20', default=20)
    parser.add_argument('--save-path', help='Where to save the output csv, default: ../data/measurements',
                        default='../data/measurements')

    args = parser.parse_args()
    pandarallel.initialize(nb_workers=args.n_cores)
    files = get_files()

    print("Calculating daily cramming at 140 and runonver at 280 before the switch per source, per language...")
    output_before = multiprocess_batches(before_daily_wrapper, files, n_cores=args.n_cores)
    before = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_before))
    before_cramming_140 = before.parallel_apply(cramming_140_wrapper, axis=1)
    before_runover_280 = before.parallel_apply(runover_280_wrapper, axis=1)
    before_cramming_140 = before_cramming_140.to_frame().rename({0: "measurement"}, axis=1)
    before_cramming_140['measure'] = 'cramming'
    before_runover_280 = before_runover_280.to_frame().rename({0: "measurement"}, axis=1)
    before_runover_280['measure'] = 'cramming'

    print("Calculating daily cramming at 140 and 280 and runover at 280 after the switch per source, per language...")
    output_after = multiprocess_batches(after_daily_wrapper, files, n_cores=args.n_cores)
    after = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_after))
    after_cramming_280 = after.parallel_apply(cramming_280_wrapper, axis=1)
    after_cramming_140 = after.parallel_apply(cramming_140_wrapper, axis=1)
    after_runover_280 = after.parallel_apply(runover_280_wrapper, axis=1)
    after_cramming_140 = after_cramming_140.to_frame().rename({0: "measurement"}, axis=1)
    after_cramming_280 = after_cramming_280.to_frame().rename({0: "measurement"}, axis=1)
    after_runover_280 = after_runover_280.to_frame().rename({0: "measurement"}, axis=1)

    print("Calculating daily fraction of tweets longer than 140 after the switch...")
    after_empirical_140 = after.parallel_apply(get_empirical_density, axis=1)
    after_empirical_140 = after_empirical_140.to_frame().rename({0: "measurement"}, axis=1)
    
    after_empirical_140['measure'] = 'empirical fraction'
    after_cramming_140['measure'] = 'cramming'
    after_cramming_280['measure'] = 'cramming'
    after_runover_280['measure'] = 'runover'
    before_cramming_140['measurement_at'] = 140
    before_runover_280['measurement_at'] = 280
    after_cramming_140['measurement_at'] = 140
    after_cramming_280['measurement_at'] = 280
    after_runover_280['measurement_at'] = 280
    after_empirical_140['measurement_at'] = 140

    df = pd.concat([before_cramming_140, before_runover_280,
                    after_cramming_140, after_cramming_280,
                     after_runover_280, after_empirical_140])

    df = df.reset_index().set_index('created_at')

    print("Saving to file...")
    df.to_csv(os.path.join(args.save_path, "daily_measures_per_lang_per_source.csv.gz"))
