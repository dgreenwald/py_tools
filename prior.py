# import math
import numpy as np
import scipy.stats as st

BETA = 1
GAMMA = 2
INV_GAMMA = 3
NORM = 4

def get_prior(prior_type, mean=None, sd=None, params=None):

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

    if params is None:
        assert (mean is not None) or (sd is not None)
        if prior_num == BETA:
            alp = (1.0 - mean) * ((mean / sd) ** 2) - mean
            bet = (1.0 - mean) * alp / mean
            return st.beta(alp, bet)
        elif prior_num == GAMMA:
            the = (sd ** 2) / mean
            k = mean / the
            return st.gamma(k, scale=the)
        elif prior_num == INV_GAMMA:
            alp = 2 + ((mean / sd) ** 2)  
            bet = mean * (alp - 1)
            return st.invgamma(alp, scale=bet)
        elif prior_num == NORM:
            return st.norm(loc=mean, scale=sd)
        else:
           raise Exception

    return (prior_num, params)

class Prior:
    """Bayesian prior"""

    def __init__(self):
        self.dists = []

    def add(self, prior_type, *args, **kwargs):
        self.dists.append(get_prior(prior_type, *args, **kwargs))

    def logpdf(self, vals):
        return np.sum([dist.logpdf(val) for dist, val in zip(self.dists, vals)])
