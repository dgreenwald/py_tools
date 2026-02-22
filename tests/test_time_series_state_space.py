import numpy as np

from py_tools.time_series.state_space import StateSpaceEstimates, StateSpaceModel


def _ssm_1d():
    A = np.array([[0.8]])
    R = np.array([[1.0]])
    Q = np.array([[0.2]])
    Z = np.array([[1.0]])
    H = np.array([[0.05]])
    b = np.array([0.0])
    return StateSpaceModel(A, R, Q, Z, H, b=b)


def test_simulate_matches_manual_recursion_without_meas_error():
    ssm = _ssm_1d()
    x1 = np.array([1.0])
    shocks = np.array([[0.0], [1.0], [-1.0]])
    meas_err = np.zeros((4, 1))
    y, x = ssm.simulate(x_1=x1, shocks=shocks, meas_err=meas_err, use_b=False)

    x_expected = np.zeros((4, 1))
    x_expected[0, 0] = 1.0
    for t in range(1, 4):
        x_expected[t, 0] = 0.8 * x_expected[t - 1, 0] + shocks[t - 1, 0]

    assert np.allclose(x, x_expected, atol=1e-12)
    assert np.allclose(y, x_expected, atol=1e-12)


def test_decompose_by_shock_reconstructs_state_path():
    ssm = _ssm_1d()
    x1 = np.array([0.5])
    shocks = np.array([[0.2], [-0.4], [0.1], [0.0]])
    _, states = ssm.simulate(x_1=x1, shocks=shocks, meas_err=np.zeros((5, 1)), use_b=False)

    shock_components, det_component = ssm.decompose_by_shock_init(shocks, x1)
    recon = det_component + np.sum(shock_components, axis=0)
    assert shock_components.shape == (1, 5, 1)
    assert np.allclose(recon, states, atol=1e-10)


def test_decompose_y_by_state_for_1d_has_zero_removed_component():
    ssm = _ssm_1d()
    states = np.array([[1.0], [0.5], [0.25]])
    y = states @ ssm.Z.T
    y_only, y_removed = ssm.decompose_y_by_state(states, y=y, start_ix=0)
    assert y_only.shape == (1, 3, 1)
    assert y_removed.shape == (1, 3, 1)
    assert np.allclose(y_only[0, :, :], y, atol=1e-12)
    assert np.allclose(y_removed, 0.0, atol=1e-12)


def test_state_space_estimates_filter_smooth_and_shock_components_smoke():
    ssm = _ssm_1d()
    x1 = np.array([0.0])
    shocks = np.array([[0.0], [0.2], [-0.1], [0.0]])
    y, _ = ssm.simulate(x_1=x1, shocks=shocks, meas_err=np.zeros((5, 1)))
    y[2, 0] = np.nan

    est = StateSpaceEstimates(ssm, y)
    est.kalman_filter()
    est.disturbance_smoother()
    est.state_smoother()
    est.shock_smoother()
    est.meas_err_smoother()

    assert est.x_pred.shape == (5, 1)
    assert est.P_pred.shape == (5, 1, 1)
    assert est.x_smooth.shape == (5, 1)
    assert est.shocks_smooth.shape == (4, 1)
    assert est.meas_err_smooth.shape == (5, 1)

    shock_components, det_component = est.get_shock_components()
    assert shock_components.shape[0] == ssm.Ne
    assert det_component.shape == (5, 1)
