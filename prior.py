# import math
import numpy as np
from scipy import stats as st, optimize as opt

BETA = 1
GAMMA = 2
INV_GAMMA = 3
NORM = 4

def get_prior(prior_type, mean=None, std=None, mode=None, params=None):

    prior_num_dict = {
        'beta' : BETA,
        'gamma' : GAMMA,
        'inv_gamma' : INV_GAMMA,
        'norm' : NORM,
    }

    if prior_type is None:
        return None
    elif isinstance(prior_type, int):
        prior_num = prior_type
    elif isinstance(prior_type, str):
        prior_num = prior_num_dict[prior_type]
    else:
        print("Bad prior type")
        raise Exception

    if params is None:
        assert ((mean is not None) or (mode is not None))
        assert ((mean is None) or (mode is None))
        assert (std is not None)
        if prior_num == BETA:
            alp = (1.0 - mean) * ((mean / std) ** 2) - mean
            bet = (1.0 - mean) * alp / mean
            return st.beta(alp, bet)
        elif prior_num == GAMMA:
            the = (std ** 2) / mean
            k = mean / the
            return st.gamma(k, scale=the)
        elif prior_num == INV_GAMMA: 
            if mean is not None:
                alp = 2.0 + ((mean / std) ** 2) 
                bet = mean * (alp - 1.0)
            else:
                raise Exception # Not working
                def objfcn(alp, mode, std):
                    print('alp: {}'.format(alp))
                    if alp < 2.0:
                        return -1e+10 + alp
                    bet = mode * (alp + 1.0)
                    var_implied = ((bet / (alp - 1.0)) ** 2) / (alp - 2.0)
                    std_implied = np.sqrt(var_implied)
                    dist = std - std_implied
                    print('std_implied: {}'.format(std_implied))
                    return dist
                
                alp0 = 5.0
#                alp = opt.newton(objfcn, alp0, args=(mode, std))
                res = opt.root(objfcn, alp0, args=(mode, std))
                print(res)
                assert (res.success)
                alp = res.x
#                print(alp)
                bet = mode * (alp + 1.0)
            return st.invgamma(alp, scale=bet)
        elif prior_num == NORM:
            return st.norm(loc=mean, scale=std)
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
        return np.sum([dist.logpdf(val) for dist, val in zip(self.dists, vals) if dist is not None])
