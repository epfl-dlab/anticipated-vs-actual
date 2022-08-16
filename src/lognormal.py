import numpy as np

def lognormal_func(x, mu, sigma) :
    return 1 / (np.sqrt(2 * np.pi) * sigma * x) * np.exp(-((np.log(x) - mu)**2) \
                                                         / (2 * sigma**2))