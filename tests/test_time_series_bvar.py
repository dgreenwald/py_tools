import numpy as np
import pandas as pd

from py_tools.time_series import bvar as bvm


def test_xtx_and_log_abs_det():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(bvm.xtx(x), x.T @ x)

    A = np.diag([2.0, 3.0])
    assert np.isclose(bvm.log_abs_det(A), np.log(6.0))


def test_glp_hyperprior_outputs_have_expected_shapes():
    gshape, gscale, ishape, iscale = bvm.glp_hyperprior(Ny=3)
    assert gshape.shape == (3,)
    assert gscale.shape == (3,)
    assert ishape.shape == (3,)
    assert iscale.shape == (3,)
    assert np.all(gshape > 1.0)
    assert np.all(gscale > 0.0)


def test_mniw_and_dummy_prior_builders_shapes():
    Ny = 2
    p = 2
    Nx = Ny * p + 1
    params = np.array([0.5, 1.0, 2.0])
    b_bar, Om_inv_bar, df_bar = bvm.mniw_prior(params, Ny=Ny, Nx=Nx, p=p)
    assert b_bar.shape == (Ny * Nx,)
    assert Om_inv_bar.shape == (Nx, Nx)
    assert np.isclose(np.diagonal(Om_inv_bar)[-1], 1e-6)
    assert df_bar == Ny + 2

    X_star, Y_star = bvm.co_persistence_prior(
        np.array([0.8, 1.2]), Nx, Ny, p, ybar=np.array([1.0, 2.0])
    )
    assert X_star.shape == (Ny + 1, Nx)
    assert Y_star.shape == (Ny + 1, Ny)

    lam = np.array([0.2, 1.0, 1, 1.5, 2.0, 0.3, 0.2, 0.1])
    rwlist = np.array([1.0, 0.0])
    ybar = np.array([1.0, 2.0])
    sbar = np.array([0.5, 0.8])
    X_mn, Y_mn = bvm.mn_prior(lam, Nx, Ny, p, rwlist, ybar, sbar)
    assert X_mn.shape[1] == Nx
    assert Y_mn.shape[1] == Ny


def test_check_nan_var_and_compute_irfs():
    da = ["y", "x"]
    assert bvm.check_nan_var("L1_y", da) is False
    assert bvm.check_nan_var("x", da) is False
    assert bvm.check_nan_var("z", da) is True

    # p=1, Ny=1 helper IRF should follow geometric recursion.
    B = np.array([[0.5], [0.0]])
    impact = np.array([[1.0]])
    irf = bvm.compute_irfs(B, p=1, Nirf=5, impact=impact)
    assert irf.shape == (5, 1, 1)
    assert np.allclose(irf[:, 0, 0], np.array([1.0, 0.5, 0.25, 0.125, 0.0625]))


def test_bvar_class_fit_sample_and_irf_smoke():
    rng = np.random.default_rng(123)
    n = 80
    y = np.zeros((n, 2))
    eps = rng.normal(scale=0.2, size=(n, 2))
    for t in range(1, n):
        y[t, 0] = 0.5 * y[t - 1, 0] + 0.1 * y[t - 1, 1] + eps[t, 0]
        y[t, 1] = 0.2 * y[t - 1, 0] + 0.6 * y[t - 1, 1] + eps[t, 1]

    idx = pd.date_range("2000-01-01", periods=n, freq=pd.offsets.QuarterEnd())
    df = pd.DataFrame({"y1": y[:, 0], "y2": y[:, 1]}, index=idx)

    model = bvm.BVAR(df, y_vars=["y1", "y2"], p=1, glp_prior=True)
    model.add_prior()
    model.fit()
    assert model.B_hat.shape == (model.Nx, model.Ny)
    assert model.Psi_hat.shape == (model.Ny, model.Ny)

    model.sample(Nsim=6)
    assert model.B_sim.shape == (6, model.Nx, model.Ny)
    assert model.Sig_sim.shape == (6, model.Ny, model.Ny)

    model.compute_irfs_sim(Nirf=5, impact=np.eye(model.Ny))
    assert model.irf_sim.shape == (6, 5, model.Ny, model.Ny)
