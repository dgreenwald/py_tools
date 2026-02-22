import numpy as np
import pandas as pd

from py_tools.time_series import var as tsv


def test_companion_form_with_constant_has_expected_blocks():
    # Ny = 2, Nlags = 2, + constant => Nx = 5
    A = np.array(
        [
            [0.5, 0.1, 0.2, 0.0, 1.0],
            [0.0, 0.6, 0.1, 0.3, -0.5],
        ]
    )
    A_comp = tsv.companion_form(A, use_const=True)
    assert A_comp.shape == (5, 5)
    assert np.allclose(A_comp[:2, :], A)
    assert np.allclose(A_comp[2:4, 0:2], np.eye(2))
    assert A_comp[-1, -1] == 1.0


def test_compute_irfs_scalar_matches_known_recursion():
    # AR(1) with constant column included in A, but no shock loading on constant state.
    A = np.array([[0.5, 0.0]])
    B = np.array([[1.0]])
    irf = tsv.compute_irfs(A, B, Nt_irf=5)
    expected = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    assert irf.shape == (1, 1, 5)
    assert np.allclose(irf[0, 0, :], expected, atol=1e-12)


def test_var_class_fit_irf_and_bootstrap_smoke():
    rng = np.random.default_rng(42)
    n = 80
    e = rng.normal(scale=0.1, size=(n, 2))
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t, 0] = 0.7 * y[t - 1, 0] + 0.1 * y[t - 1, 1] + e[t, 0]
        y[t, 1] = 0.2 * y[t - 1, 0] + 0.6 * y[t - 1, 1] + e[t, 1]

    idx = pd.date_range("2000-01-01", periods=n, freq="QE")
    df = pd.DataFrame({"y1": y[:, 0], "y2": y[:, 1]}, index=idx)

    model = tsv.VAR(df, ["y1", "y2"], n_var_lags=1, use_const=True)
    model.fit()
    assert model.A.shape == (2, 3)
    assert model.resid.shape[1] == 2

    model.compute_irfs(Nt_irf=6)
    assert model.irfs.shape == (2, 2, 6)

    model.wild_bootstrap(Nboot=8)
    assert model.A_boot.shape == (2, 3, 8)
    assert model.y_boot.shape[2] == 8

    model.bootstrap_irfs(np.eye(2), Nt_irf=6)
    assert model.irfs_boot.shape == (2, 2, 6, 8)
