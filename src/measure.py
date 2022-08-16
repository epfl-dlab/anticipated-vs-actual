import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import lognorm

from constants import cramming_threshold_after, cramming_threshold_before
from lognormal import lognormal_func


def cramming_140_wrapper(day_hist):
    return measure(day_hist=day_hist, measurement_at=140, measure='cramming')


def cramming_280_wrapper(day_hist):
    return measure(day_hist=day_hist, measurement_at=280, measure='cramming')


def measure(day_hist, measurement_at, measure='runover', probabilities=[0.95]):
    limit = 280
    cramming_start = cramming_threshold_after
    if measurement_at == 140:
        limit = 140
        cramming_start = cramming_threshold_before
    # fill missing values
    day_hist = day_hist.hist_chars.sort_index()[:limit]\
        .reindex(index=range(1, limit+1), fill_value=0).fillna(0)
    density = (day_hist / np.sum(day_hist))
    # fit lognormal
    if density.sum() == 0.0:
        return np.nan
    try:
        mu, sigma = curve_fit(lognormal_func, range(1, limit+1), density)[0]
    except RuntimeError:
        print("Couldn't fit the model, max function call exceeded")
        return np.nan
    if measure == 'runover':
        return 1 - lognorm.cdf(limit, s=sigma, scale=np.exp(mu), loc=0)
    elif measure == 'cramming':
        lognormal = [lognormal_func(x, mu, sigma) for x in range(1, limit+1)]
        cramming = np.abs(np.sum(density.values[cramming_start:] - lognormal[cramming_start:]))
        return cramming
    elif measure == 'both':
        runover = 1 - lognorm.cdf(limit, s=sigma, scale=np.exp(mu), loc=0)
        lognormal = [lognormal_func(x, mu, sigma) for x in range(1, limit+1)]
        cramming = np.abs(np.sum(density.values[cramming_start:] - lognormal[cramming_start:]))
        return runover, cramming
    elif measure == 'num_chars':
        return [lognorm.ppf(q=probability, s=sigma, scale=np.exp(mu), loc=0) for probability in probabilities]
    else:
        raise ValueError("Unknown measure")

