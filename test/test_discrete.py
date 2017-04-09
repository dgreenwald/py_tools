import numpy as np
from py_tools.discrete import DiscreteModel
import py_tools.econ as ec

R = 1.0 / 0.99
bet = 0.98
rho_z = 0.95
sig_z = 0.1
Nz = 3

Nx = 5
x_grid = np.linspace(0.0, 5.0, Nx)
z_grid, Pz = ec.discrete_approx(rho_z, sig_z, Nz)

xz_stack = np.transpose([
    np.repeat(x_grid, len(z_grid)),
    np.tile(z_grid, len(x_grid)), 
])

flow_list = []
for iz in range(Nz):
    z = z_grid[iz]
    c = np.exp(z) + x_grid[:, np.newaxis] - (x_grid[np.newaxis, :] / R)

    flow_i = np.zeros((Nx, Nx))
    flow_i[c > 0] = np.log(c[c > 0])
    flow_i[c <= 0] = -np.inf
    flow_list.append(flow_i)

# Nsim = 1000

dm = DiscreteModel(bet, flow_list, x_grid[:, np.newaxis], z_grid[:, np.newaxis], Pz)
dm.solve()
# xsim, zsim = dm.sim(Nsim))
