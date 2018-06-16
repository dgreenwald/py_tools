import math
import numpy as np
import scipy.stats

BETA = 1
GAMMA = 2
INV_GAMMA = 3
NORM = 4

# def set_beta(mean=None, mode=None, sd=None, var=None):

    # assert (mean is None or mode is None)
    # assert (sd is None or var is None)

    # if var is None:
        # var = sd ** 2

    # if mean is not None:
        # alp = ((1.0 - mean) * (mean ** 2) / var) - mean
        # bet = (1.0 - mean) * alp / mean


def set_prior(prior_type, params=None, mean=None, sd=None):

    prior_num_dict = {
        'beta' : BETA,
        'gamma' : GAMMA,
        'inv_gamma' : INV_GAMMA,
        'norm' : NORM,
    }

    if isinstance(prior_type, int):
        prior_num = prior_type
    elif isinstance(prior_type, str):
        prior_num = prior_num_dict[prior_type]
    else:
        print("Bad prior type")
        raise Exception

   assert (not (params is None and (mean is None or sd is None)))

   if params is None:
       if prior_num == BETA:
            alp = ((1.0 - mean) * (mean ** 2) / var) - mean
            bet = (1.0 - mean) * alp / mean
            params = np.array((alp, bet))
        elif prior_num == NORM:
            params = np.array((mean, sd))
        else:
            raise Exception

    return (prior_num, params)


def eval_prior(params, prior_list):

    L = 0.0

    for ii, (prior_num, prior_params) in enumerate(prior_list):
        if prior_num is not None:
            if prior_num == BETA:
                L += stats.beta.logpdf(val, prior_params[0], prior_params[1])
            elif prior_num == NORM:
                L += stats.norm.logpdf(val, prior_params[0], prior_params[1])

    return L
