"""Tests for py_tools.econ.discrete"""

import numpy as np
import pytest

from py_tools.econ.discrete import (
    to_2d,
    combine_grids,
    combine_grids_from_list,
    combine_markov_chains,
    drop_low_probs,
    DiscreteModel,
)


# --- to_2d ---

class TestTo2d:
    def test_1d_becomes_2d(self):
        x = np.array([1.0, 2.0, 3.0])
        result = to_2d(x)
        assert result.shape == (3, 1)

    def test_2d_unchanged(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = to_2d(x)
        assert result.shape == (2, 2)
        assert np.array_equal(result, x)

    def test_3d_returns_none(self):
        x = np.ones((2, 3, 4))
        result = to_2d(x)
        assert result is None


# --- combine_grids ---

class TestCombineGrids:
    def test_output_shape(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0, 5.0])
        result = combine_grids(x, y)
        # Should be (len(x)*len(y), 2)
        assert result.shape == (6, 2)

    def test_cartesian_product(self):
        x = np.array([1.0, 2.0])
        y = np.array([10.0, 20.0])
        result = combine_grids(x, y)
        assert result.shape == (4, 2)
        # First column: [1, 1, 2, 2]
        assert np.array_equal(result[:, 0], [1.0, 1.0, 2.0, 2.0])
        # Second column: [10, 20, 10, 20]
        assert np.array_equal(result[:, 1], [10.0, 20.0, 10.0, 20.0])

    def test_2d_input(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 rows, 2 cols
        y = np.array([5.0, 6.0, 7.0])
        result = combine_grids(x, y)
        assert result.shape == (6, 3)


# --- combine_grids_from_list ---

class TestCombineGridsFromList:
    def test_two_grids(self):
        g1 = np.array([1.0, 2.0])
        g2 = np.array([3.0, 4.0, 5.0])
        result = combine_grids_from_list([g1, g2])
        assert result.shape == (6, 2)

    def test_three_grids(self):
        g1 = np.array([0.0, 1.0])
        g2 = np.array([0.0, 1.0])
        g3 = np.array([0.0, 1.0])
        result = combine_grids_from_list([g1, g2, g3])
        # 2*2*2 = 8 rows, 3 cols
        assert result.shape == (8, 3)


# --- combine_markov_chains ---

class TestCombineMarkovChains:
    def test_kron_transition(self):
        # Two 2-state chains
        P1 = np.array([[0.7, 0.3], [0.2, 0.8]])
        P2 = np.array([[0.6, 0.4], [0.1, 0.9]])
        g1 = np.array([0.0, 1.0])
        g2 = np.array([0.0, 1.0])
        grid, P = combine_markov_chains([g1, g2], [P1, P2])
        assert P.shape == (4, 4)
        assert grid.shape == (4, 2)

    def test_kron_rows_sum_to_one(self):
        P1 = np.array([[0.7, 0.3], [0.2, 0.8]])
        P2 = np.array([[0.5, 0.5], [0.4, 0.6]])
        g1 = np.array([0.0, 1.0])
        g2 = np.array([0.0, 1.0])
        _, P = combine_markov_chains([g1, g2], [P1, P2])
        assert np.allclose(P.sum(axis=1), 1.0)


# --- drop_low_probs ---

class TestDropLowProbs:
    def test_zeros_small_probs(self):
        P = np.array([[0.9, 1e-7, 0.1 - 1e-7],
                      [0.5, 0.5, 0.0]])
        P_clean = drop_low_probs(P, tol=1e-6)
        # The 1e-7 entry should become 0
        assert P_clean[0, 1] == 0.0

    def test_rows_still_sum_to_one(self):
        P = np.array([[0.5, 1e-8, 0.5],
                      [0.3, 0.7, 0.0]])
        P_clean = drop_low_probs(P, tol=1e-6)
        assert np.allclose(P_clean.sum(axis=1), 1.0)

    def test_no_change_above_tol(self):
        P = np.array([[0.6, 0.4],
                      [0.3, 0.7]])
        P_clean = drop_low_probs(P, tol=1e-6)
        assert np.allclose(P_clean.sum(axis=1), 1.0)
        # All entries are above tol, rows should still be normalized
        assert np.allclose(P_clean, P / P.sum(axis=1)[:, np.newaxis])


# --- DiscreteModel ---

class TestDiscreteModel:
    """Simple 2-state, 1-shock discrete dynamic programming test.

    State space: x in {0, 1}.
    Shock space: z in {0, 1}.
    Flow payoff: u(x, x') = x' (reward is next-period state).
    Discount: beta = 0.9 (same for both z-states).
    Transition: iid shocks, Pz = [[0.5, 0.5], [0.5, 0.5]].
    Optimal policy: always choose x'=1 regardless of current x or z.
    """

    def _make_model(self):
        # Flow: u[x, x'] = x' for both z states => (2,2) array
        flow = np.array([[0.0, 1.0],
                         [0.0, 1.0]])
        flow_list = [flow, flow]
        x_grid = np.array([[0.0], [1.0]])
        z_grid = np.array([[0.0], [1.0]])
        Pz = np.array([[0.5, 0.5], [0.5, 0.5]])
        bet = np.array([0.9, 0.9])
        return DiscreteModel(bet, flow_list, x_grid, z_grid, Pz)

    def test_construction(self):
        model = self._make_model()
        assert model.Nx == 2
        assert model.Nz == 2

    def test_solve_converges(self, capsys):
        model = self._make_model()
        model.solve()
        assert hasattr(model, 'V')
        assert hasattr(model, 'I')

    def test_optimal_policy_is_one(self):
        model = self._make_model()
        model.solve()
        # Policy should always be to choose x'=1 (index 1)
        assert np.all(model.I == 1)

    def test_value_shape(self):
        model = self._make_model()
        model.solve()
        assert model.V.shape == (model.Nz, model.Nx)

    def test_get_opt_flow_shape(self):
        model = self._make_model()
        opt_flow = model.get_opt_flow()
        assert opt_flow.shape == (model.Nz * model.Nx, 1)

    def test_get_P_trans_shape(self):
        model = self._make_model()
        P_trans = model.get_P_trans(discount=False, sparse=True)
        import scipy.sparse as sp
        assert sp.issparse(P_trans)
        assert P_trans.shape == (model.Nz * model.Nx, model.Nz * model.Nx)

    def test_finite_values(self):
        model = self._make_model()
        model.solve()
        assert np.all(np.isfinite(model.V))
