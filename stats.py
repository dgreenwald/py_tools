import numpy as np

def weighted_quantile(values, weights, quantiles, sort=True):

    if sort:
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]
    
    cumulative_weights = np.cumsum(weights) - weights[0]
    cumulative_weights /= cumulative_weights[-1]

    return np.interp(quantiles, cumulative_weights, values)
