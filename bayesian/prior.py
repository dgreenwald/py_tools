import numpy as np
from scipy import stats as st, optimize as opt

BETA = 1
GAMMA = 2
INV_GAMMA = 3
NORM = 4
TRUNC_NORM = 5

def get_prior(prior_type, mean=None, sd=None):

    prior_num_dict = {
        'beta' : BETA,
        'gamma' : GAMMA,
        'inv_gamma' : INV_GAMMA,
        'norm' : NORM,
        'trunc_norm' : TRUNC_NORM,
    }

    if prior_type is None:
        return None
    elif isinstance(prior_type, int):
        prior_num = prior_type
    elif isinstance(prior_type, str):
        if prior_type not in prior_num_dict:
            raise ValueError("Unknown prior type: {}".format(prior_type))
        prior_num = prior_num_dict[prior_type]
    else:
        raise TypeError(
            "prior_type must be None, int, or str; got {}".format(type(prior_type).__name__)
        )

    if mean is None or sd is None:
        raise ValueError(
            "mean and sd must both be provided for prior_type={}".format(prior_type)
        )

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
    elif prior_num == TRUNC_NORM:
        a = (0.0 - mean) / sd
        b = (1.0 - mean) / sd
        return st.truncnorm(a, b, mean, sd)
    else:
        raise RuntimeError("Unexpected prior code")

class Prior:
    """Bayesian prior"""

    def __init__(self):
        self.dists = []
        self.names = []
        self.non_flat_names = []

    def add(self, prior_type, name=None, *args, **kwargs):
        this_prior = get_prior(prior_type, *args, **kwargs)
        self.dists.append(this_prior)

        # Add parameter name
        if name is None:
            name = 'param{:d}'.format(len(self.dists))
        self.names.append(name)
        
        if this_prior is not None:
            self.non_flat_names.append(name)

    def logpdf(self, vals):
        logpdf_list = [dist.logpdf(val) for dist, val in zip(self.dists, vals) if dist is not None]
        if logpdf_list:
            return np.sum(logpdf_list)
        else:
            return 0.0

    def sample(self, n_samp):
        if any(dist is None for dist in self.dists):
            raise ValueError("Cannot sample from Prior with flat (None) components.")
        return np.vstack([dist.rvs(n_samp) for dist in self.dists])
