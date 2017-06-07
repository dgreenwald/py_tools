import numpy as np
from py_tools.discrete import DiscreteModel
import py_tools.econ as ec
import pyviennacl as p
from argmax2 import argmax

from colloc import in_out

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

flow = np.vstack(flow_list)

V = np.zeros((Nx, Nz))

# To save files, use syntax below (replacing path)
# in_out.saveEigen(flow, '/home/dan/Dropbox/sandbox/discrete/flow.out')
# in_out.saveEigen(Pz, '/home/dan/Dropbox/sandbox/discrete/Pz.out')
# in_out.saveEigen(x_grid[:, np.newaxis], '/home/dan/Dropbox/sandbox/discrete/x_grid.out')
# in_out.saveEigen(z_grid[:, np.newaxis], '/home/dan/Dropbox/sandbox/discrete/z_grid.out')

# Load these in Eigen using <helper.hpp> and the syntax e.g.

# MatXd Pz;
# loadEigen("/path/to/Pz.out", Pz)

# Also works with other Eigen types (ArrXXd, VecXd, ArrXd).

# TEST WITH ORIGINAL VERSION
dm = DiscreteModel(bet, flow_list, x_grid[:, np.newaxis], z_grid[:, np.newaxis], Pz)
dm.solve()

# ALGORITHM HERE

# Inputs should be:

# bet: scalar
# flow_list: (Nx * Nz) x Nx matrix
# Pz: Nz x Nz matrix
# V: Nx x Nz matrix

# Outputs should be:

# V (see above)
# indices (Nx * Nz) vector of integers

Ns = Nx * Nz

bet_Pz = bet * Pz # pre-compute
A = np.zeros((Ns, Ns))
f_star = np.zeros(Ns)
indices = (-1 * np.ones(Ns)).astype(int)
done = False
tag = p.gmres_tag(tolerance = 1e-5, max_iterations = 150, krylov_dim= 50)
while not done:

    old_indices = indices.copy()

    #should this be run with Vienna?
    Om = bet * np.dot(V, Pz.T) # expected future value
    for jj in range(Nz):

        E = np.zeros((Nx, Nx)) # storage for "picker" matrix

        for ii in range(Nx):

            #should this be run with vienna
            kk = Nx * jj + ii
            alternatives = flow[kk, :] + Om[:, jj] # values for possible choices of i
            max_ix = argmax(alternatives, 3) # optimal choice of i
            max_val = flow[kk, max_ix] # value associated with optimal choice

            indices[kk] = max_ix # store optimal index
            f_star[kk] = max_val # store optimal value
            E[ii, max_ix] = 1.0 # put 1 at index in picker matrix
    
        A[Nx * jj : Nx * (jj+1), :] = np.kron(bet_Pz[jj, :], E) # fill in block of A matrix
        
    #hangs with upper_tag solver, so which to use?
    A_p = p.Matrix(np.eye(Ns) - A)
    b_p = p.Vector(f_star)
    #v = np.linalg.solve(np.eye(Ns)-A, f_star)
    v_p = p.linalg.solve(A_p, b_p, tag) # solve for new values if current policy forever
    #print(v)
    #print(v_p)
    v = v_p.value
    V = v.reshape((Nx, Nz), order='F') # stack colwise into Nx x Nz matrix

    done = np.all(indices == old_indices)
