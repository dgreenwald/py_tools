"""Tests for py_tools.econ.core"""

import numpy as np
import pytest

from py_tools.econ.core import (
    get_unit_vecs,
    stationary_doubling,
    ergodic_dist,
    check_ergodic,
    markov_std,
    update_value,
    get_transition,
    sim_discrete,
    sim_iid,
    sim_discrete_from_ergodic,
    multi_choice,
    sim_policy,
    sim_life_cycle,
    sim_ar1,
    sim_ar1_multi,
    sim_cir,
    discrete_approx,
)


# ---- Simple 2-state Markov chain fixture ----
# P[i,j] = prob of going from i to j
# Stationary distribution: [0.4, 0.6]
@pytest.fixture
def two_state_chain():
    P = np.array([[0.7, 0.3],
                  [0.2, 0.8]])
    pi_star = np.array([0.4, 0.6])
    return P, pi_star


# --- get_unit_vecs ---

class TestGetUnitVecs:
    def test_symmetric_chain(self, two_state_chain):
        P, pi_star = two_state_chain
        vecs = get_unit_vecs(P.T, normalize=True)
        assert vecs.shape[1] >= 1
        # Should recover the stationary distribution (up to sign)
        recovered = np.abs(vecs[:, 0])
        assert np.allclose(recovered, pi_star, atol=1e-6)

    def test_without_normalize(self, two_state_chain):
        P, _ = two_state_chain
        vecs = get_unit_vecs(P.T, normalize=False)
        # Eigenvalue should be 1
        vals, _ = np.linalg.eig(P.T)
        unit = np.abs(vals - 1.0) < 1e-8
        assert unit.sum() == vecs.shape[1]

    def test_returns_real(self, two_state_chain):
        P, _ = two_state_chain
        vecs = get_unit_vecs(P.T, normalize=True)
        assert np.isrealobj(vecs)


# --- stationary_doubling ---

class TestStationaryDoubling:
    def test_known_stationary(self, two_state_chain):
        P, pi_star = two_state_chain
        pi = stationary_doubling(P)
        assert np.allclose(pi, pi_star, atol=1e-8)

    def test_sums_to_one(self, two_state_chain):
        P, _ = two_state_chain
        pi = stationary_doubling(P)
        assert np.isclose(np.sum(pi), 1.0)

    def test_invariant(self, two_state_chain):
        P, _ = two_state_chain
        pi = stationary_doubling(P)
        assert np.allclose(pi @ P, pi, atol=1e-8)

    def test_no_convergence_raises(self):
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        # This should converge fine (all-same rows), not raise
        pi = stationary_doubling(P, maxit=500)
        assert np.allclose(pi, [0.5, 0.5], atol=1e-8)

    def test_with_seed(self, two_state_chain):
        P, pi_star = two_state_chain
        pi = stationary_doubling(P, pi_seed=np.array([0.5, 0.5]))
        assert np.allclose(pi, pi_star, atol=1e-8)


# --- ergodic_dist ---

class TestErgodic:
    def test_eigenvalue_method(self, two_state_chain):
        P, pi_star = two_state_chain
        pi = ergodic_dist(P)
        # get_unit_vecs normalizes, result is a column vector
        assert np.allclose(np.abs(pi.ravel()), pi_star, atol=1e-6)

    def test_doubling_method(self, two_state_chain):
        P, pi_star = two_state_chain
        pi = ergodic_dist(P, doubling=True)
        assert np.allclose(pi, pi_star, atol=1e-8)


# --- check_ergodic ---

class TestCheckErgodic:
    def test_ergodic_matrix(self):
        # All rows identical => ergodic
        row = np.array([[0.4, 0.6]])
        invariant = np.repeat(row, 3, axis=0)
        assert check_ergodic(invariant)

    def test_non_ergodic_matrix(self):
        invariant = np.array([[0.4, 0.6], [0.5, 0.5], [0.4, 0.6]])
        assert not check_ergodic(invariant)


# --- markov_std ---

class TestMarkovStd:
    def test_deterministic_chain(self):
        # Absorbing state: always stay at val[0]
        P = np.array([[1.0, 0.0], [1.0, 0.0]])
        vals = np.array([2.0, 5.0])
        sig = markov_std(P, vals)
        # E[v] = 2, E[v^2] = 4, Var = 0
        assert np.allclose(sig, [0.0, 0.0], atol=1e-10)

    def test_known_values(self):
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        vals = np.array([0.0, 2.0])
        sig = markov_std(P, vals)
        # E[v] = 1, E[v^2] = 2, Var = 1
        assert np.allclose(sig, [1.0, 1.0], atol=1e-10)


# --- update_value ---

class TestUpdateValue:
    def test_selects_max(self):
        V = np.array([[1.0, 3.0, 2.0],
                      [4.0, 0.0, 1.0]])
        indices, v = update_value(V)
        assert np.array_equal(indices, [1, 0])
        assert np.allclose(v, [3.0, 4.0])

    def test_shape(self):
        V = np.ones((5, 4))
        indices, v = update_value(V)
        assert indices.shape == (5,)
        assert v.shape == (5,)


# --- get_transition ---

class TestGetTransition:
    def test_dense_output(self):
        indices = np.array([1, 0, 2])
        T = get_transition(indices, sparse=False)
        assert T.shape == (3, 3)
        assert np.allclose(T[0], [0, 1, 0])
        assert np.allclose(T[1], [1, 0, 0])
        assert np.allclose(T[2], [0, 0, 1])

    def test_row_sums_to_one(self):
        indices = np.array([2, 1, 0, 2])
        T = get_transition(indices, sparse=False)
        assert np.allclose(T.sum(axis=1), 1.0)

    def test_sparse_output(self):
        indices = np.array([1, 0])
        T = get_transition(indices, sparse=True)
        import scipy.sparse as sp
        assert sp.issparse(T)


# --- sim_discrete ---

class TestSimDiscrete:
    def test_output_length(self, two_state_chain):
        P, _ = two_state_chain
        np.random.seed(42)
        ix = sim_discrete(P, N=50)
        assert len(ix) == 50

    def test_valid_indices(self, two_state_chain):
        P, _ = two_state_chain
        np.random.seed(42)
        ix = sim_discrete(P, N=100)
        assert np.all(ix >= 0) and np.all(ix < P.shape[0])

    def test_starts_at_i0(self, two_state_chain):
        P, _ = two_state_chain
        ix = sim_discrete(P, N=10, i0=1)
        assert ix[0] == 1


# --- sim_iid ---

class TestSimIid:
    def test_output_length(self):
        p = np.array([0.3, 0.7])
        np.random.seed(0)
        samples = sim_iid(p, 200)
        assert len(samples) == 200

    def test_valid_indices(self):
        p = np.array([0.2, 0.3, 0.5])
        np.random.seed(0)
        samples = sim_iid(p, 100)
        assert np.all(samples >= 0) and np.all(samples < 3)

    def test_frequency_proportional(self):
        np.random.seed(0)
        p = np.array([0.1, 0.9])
        samples = sim_iid(p, 10000)
        np.random.seed(None)
        freq = np.bincount(samples, minlength=2) / 10000
        assert np.allclose(freq, p, atol=0.02)


# --- sim_discrete_from_ergodic ---

class TestSimDiscreteFromErgodic:
    def test_output_length(self, two_state_chain):
        P, pi_star = two_state_chain
        np.random.seed(0)
        ix = sim_discrete_from_ergodic(P, 30, pi_star=pi_star)
        assert len(ix) == 30

    def test_valid_indices(self, two_state_chain):
        P, pi_star = two_state_chain
        np.random.seed(0)
        ix = sim_discrete_from_ergodic(P, 50)
        assert np.all((ix >= 0) & (ix < 2))


# --- multi_choice ---

class TestMultiChoice:
    def test_output_shape(self):
        np.random.seed(0)
        p = np.array([[0.2, 0.8], [0.6, 0.4]])
        choices = multi_choice(p)
        assert choices.shape == (2,)

    def test_valid_choices(self):
        np.random.seed(0)
        p = np.ones((10, 3)) / 3.0
        choices = multi_choice(p)
        assert np.all((choices >= 0) & (choices < 3))

    def test_deterministic_choice(self):
        # All probability on index 1
        p = np.array([[0.0, 1.0, 0.0],
                      [0.0, 1.0, 0.0]])
        choices = multi_choice(p)
        assert np.all(choices == 1)


# --- sim_policy ---

class TestSimPolicy:
    def test_identity_policy(self):
        # Policy: always stay at current x
        index_list = [np.array([0, 1, 2]),  # z=0: x->x
                      np.array([0, 1, 2])]  # z=1: x->x
        z_ix_sim = np.array([0, 1, 0, 1])
        ix = sim_policy(index_list, z_ix_sim, i0=2)
        # With identity policy, x stays at i0=2
        assert np.all(ix == 2)

    def test_output_length(self):
        index_list = [np.array([1, 0]), np.array([0, 1])]
        z_ix_sim = np.array([0, 0, 1, 1])
        ix = sim_policy(index_list, z_ix_sim)
        assert len(ix) == 4


# --- sim_life_cycle ---

class TestSimLifeCycle:
    def test_output_length(self):
        # 3-period life cycle with identity policy
        index_lists = [
            [np.array([0, 1]), np.array([0, 1])],  # t=0
            [np.array([0, 1]), np.array([0, 1])],  # t=1
            [np.array([0, 1]), np.array([0, 1])],  # t=2
        ]
        z_ix_sim = np.array([0, 1, 0])
        ix = sim_life_cycle(index_lists, z_ix_sim, i0=1)
        assert len(ix) == 3

    def test_identity_policy(self):
        index_lists = [
            [np.array([0, 1]), np.array([0, 1])],
            [np.array([0, 1]), np.array([0, 1])],
        ]
        z_ix_sim = np.array([0, 1])
        ix = sim_life_cycle(index_lists, z_ix_sim, i0=1)
        assert np.all(ix == 1)


# --- sim_ar1 ---

class TestSimAr1:
    def test_output_length(self):
        x = sim_ar1(0.9, 0.1, Nsim=50)
        assert len(x) == 50

    def test_with_fixed_errors(self):
        # With zero shocks and mu=2.0, all values equal mu regardless of rho
        e = np.zeros(5)
        x = sim_ar1(0.9, 0.1, mu=2.0, Nsim=5, e=e)
        assert np.allclose(x, 2.0, atol=1e-10)

    def test_with_x0(self):
        e = np.zeros(3)
        x = sim_ar1(0.5, 0.1, mu=0.0, Nsim=3, e=e, x0=4.0)
        assert np.isclose(x[0], 4.0)
        assert np.isclose(x[1], 2.0)
        assert np.isclose(x[2], 1.0)

    def test_mean_for_long_series(self):
        np.random.seed(42)
        x = sim_ar1(0.5, 0.2, mu=3.0, Nsim=10000)
        np.random.seed(None)
        assert np.isclose(np.mean(x), 3.0, atol=0.1)


# --- sim_ar1_multi ---

class TestSimAr1Multi:
    def test_output_shape(self):
        x = sim_ar1_multi(0.9, 0.1, Nper=20, Nsim=5)
        assert x.shape == (5, 20)

    def test_with_fixed_errors(self):
        e = np.zeros((3, 4))
        x = sim_ar1_multi(0.5, 0.1, Nper=4, Nsim=3, e=e, mu=2.0)
        assert np.allclose(x, 2.0, atol=1e-10)


# --- sim_cir ---

class TestSimCir:
    def test_requires_x0(self):
        with pytest.raises(Exception):
            sim_cir(0.9, 0.1, mu=1.0, Nsim=10, x0=None)

    def test_output_length(self):
        x = sim_cir(0.9, 0.1, mu=1.0, Nsim=20, x0=1.0)
        assert len(x) == 20

    def test_nonnegative_with_bound(self):
        np.random.seed(0)
        x = sim_cir(0.5, 2.0, mu=1.0, Nsim=100, x0=0.5, bound=True)
        assert np.all(x >= 0)

    def test_starting_value(self):
        x = sim_cir(0.9, 0.1, mu=1.0, Nsim=10, x0=2.5)
        assert np.isclose(x[0], 2.5)


# --- discrete_approx (Rouwenhorst) ---

class TestDiscreteApprox:
    def test_output_sizes(self):
        y, P = discrete_approx(0.9, 0.1, N=5)
        assert len(y) == 5
        assert P.shape == (5, 5)

    def test_row_sums_to_one(self):
        _, P = discrete_approx(0.9, 0.1, N=7)
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_nonnegative_probabilities(self):
        _, P = discrete_approx(0.8, 0.2, N=4)
        assert np.all(P >= 0)

    def test_symmetric_grid(self):
        # Without constant shift, grid should be symmetric around 0
        y, _ = discrete_approx(0.7, 0.15, N=5)
        assert np.isclose(y[0], -y[-1])

    def test_constant_shift(self):
        cons = 1.5
        y_shifted, _ = discrete_approx(0.7, 0.15, N=5, cons=cons)
        y_base, _ = discrete_approx(0.7, 0.15, N=5, cons=0.0)
        assert np.allclose(y_shifted, y_base + cons)

    def test_two_state(self):
        # For N=2: q = (1+rho)/2
        rho = 0.6
        _, P = discrete_approx(rho, 0.1, N=2)
        q = 0.5 * (1.0 + rho)
        expected = np.array([[q, 1 - q], [1 - q, q]])
        assert np.allclose(P, expected, atol=1e-10)

    def test_stationary_dist_symmetric(self):
        # For symmetric Rouwenhorst the stationary dist should be symmetric
        _, P = discrete_approx(0.5, 0.1, N=5)
        pi = stationary_doubling(P)
        assert np.allclose(pi, pi[::-1], atol=1e-6)
