import argparse
import os
import warnings
from functools import reduce
import pandas as pd
import numpy as np
from pandarallel import pandarallel
from scipy.optimize import OptimizeWarning

from daily_runover_vs_cramming_280_per_source import before_daily_wrapper, after_daily_wrapper
from measure import measure
from tweets_multiprocessing import get_files, multiprocess_batches
from daily_cramming import process_batch_daily
from constants import allowed_languages

warnings.filterwarnings('ignore', category=OptimizeWarning)


def before_daily_wrapper(x):
    return process_batch_daily(x, before_switch=True, langs=allowed_languages, sources="allowed")


def after_daily_wrapper(x):
    return process_batch_daily(x, before_switch=False, langs=allowed_languages, sources="allowed")


def before_num_chars_wrapper(day_hist, probabilities):
    return measure(day_hist=day_hist, measure='num_chars', measurement_at=140, probabilities=probabilities)


def after_num_chars_wrapper(day_hist, probabilities):
    return measure(day_hist=day_hist, measure='num_chars', measurement_at=280, probabilities=probabilities)

def postprocess(df, probs):
    df = pd.DataFrame(df.tolist(), index=df.index, columns=probs)
    return df.melt(var_name='probability', value_name='num_chars', ignore_index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Daily number of characters for which a fraction of tweets would be below or equal to it,'
                    ' output of the quantile function of daily fitted lognormal for the given probability, web and mobile together')

    parser.add_argument('--n-cores', type=int, help='How many cores to use with multiprocessing, default: 20',
                        default=20)
    parser.add_argument('--save-path', help='Where to save the output csv, default: ../data/measurements',
                        default='../data/measurements')
    parser.add_argument('--probabilities', nargs="+", help='List of probabilities at which to evaluate quantile function '
                                              'to obtain number of characters, default: [95%%, 96%%, 97%%, 98%%, 99%%]',
                        default=np.arange(0.95, 0.99, 0.01))

    args = parser.parse_args()
    args.probabilities = list(map(lambda x: float(x), args.probabilities))
    pandarallel.initialize(nb_workers=args.n_cores)
    files = get_files()

    print(f"Calculating daily hypothetical number of characters for which {[int(prob*100) for prob in args.probabilities]}% of "
        f"tweets are shorter or equal before the switch...")
    output_before = multiprocess_batches(before_daily_wrapper, files, n_cores=args.n_cores)
    before = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_before))
    before_num_chars = before.parallel_apply(before_num_chars_wrapper,
                                            probabilities=args.probabilities,
                                            axis=1)
    before_outputs = postprocess(before_num_chars, probs=args.probabilities)                                      

    print(f"Calculating daily hypothetical number of characters for which {[int(prob*100) for prob in args.probabilities]}% of "
      f"tweets are shorter or equal after the switch...")
    output_after = multiprocess_batches(after_daily_wrapper, files, n_cores=args.n_cores)
    after = reduce(lambda a, b: a.add(b, fill_value=0), filter(lambda x: not x.empty, output_after))
    after_num_chars = after.parallel_apply(after_num_chars_wrapper,
                                        probabilities=args.probabilities,
                                        axis=1)
    after_outputs = postprocess(after_num_chars, probs=args.probabilities)      

    df = pd.concat([before_outputs, after_outputs])
    df.index = pd.to_datetime(df.index)
    print("Saving file...")
    df.to_csv(os.path.join(args.save_path, "daily_num_chars.csv.gz"))
