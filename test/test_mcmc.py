import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from py_tools.mcmc import MCMC

def log_like(params, x):

    mu = params[0]
    sig = params[1]

    return np.sum(norm.logpdf(x, mu, sig))

names = ['mu', 'sig']
bounds_dict = {'sig' : (0.0, None)}

mu = -0.5
sig = 0.2

N = 200
data = mu + sig * np.random.randn(N)

mc = MCMC(log_like, args=(data,), bounds_dict=bounds_dict, names=names)

x0 = np.array((0.0, 0.1))

Nsim = 1000
mc.run_all(x0, Nsim, out_dir='/home/dan/Downloads',
           jump_scale=2.0, stride=10)
#mc.find_mode(x0)
#mc.compute_hessian()
#mc.sample(1000, jump_scale=2.0, stride=10)
#mc.save('/home/dan/Downloads')

print(mc.acc_rate)
fig = plt.figure()
plt.hist(mc.draws[:, 0])
plt.show()
plt.close(fig)

fig = plt.figure()
plt.hist(mc.draws[:, 1])
plt.show()
plt.close(fig)