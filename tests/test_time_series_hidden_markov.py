import numpy as np

from py_tools.time_series.hidden_markov import HiddenMarkov, make_2d


def _build_hm():
    x_grid = np.array([-1.0, 0.0, 2.0])
    P = np.array(
        [
            [0.85, 0.10, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.10, 0.85],
        ]
    )
    y_vals = np.array([[-0.3], [1.4], [0.2], [1.8]])
    sig2 = 0.25

    def log_err_density(y_t, tt):
        y = float(y_t[0])
        return -0.5 * ((y - x_grid) ** 2) / sig2

    hm = HiddenMarkov(P, log_err_density, y_vals)
    hm.init_stationary()
    hm.filter()
    hm.smooth()
    return hm, x_grid


def test_make_2d_promotes_1d_arrays():
    x = np.array([1.0, 2.0, 3.0])
    out = make_2d(x)
    assert out.shape == (1, 3)
    assert np.allclose(out[0, :], x)


def test_filter_and_smooth_probabilities_are_normalized():
    hm, _ = _build_hm()
    assert np.allclose(hm.px_filt.sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(hm.px_smooth.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(np.isfinite(hm.log_p_err))


def test_filtered_and_smoothed_vals_have_expected_shape():
    hm, x_grid = _build_hm()
    fvals = hm.filtered_vals(x_grid)
    svals = hm.smoothed_vals(x_grid)
    assert fvals.shape == (hm.Nt, 1)
    assert svals.shape == (hm.Nt, 1)


def test_smoothed_quantiles_matches_state_axis_cdf():
    hm, x_grid = _build_hm()

    # Class-produced smoothed distribution should be (time x state).
    assert hm.px_smooth.shape == (hm.Nt, len(x_grid))
    assert np.allclose(np.sum(hm.px_smooth, axis=1), 1.0, atol=1e-10)

    q = np.array([0.25, 0.50, 0.75])
    q_vals = hm.smoothed_quantiles(x_grid, q)

    # Reference quantiles: CDF over states within each time period.
    q_ref = np.zeros((len(q), 1, hm.Nt))
    for tt in range(hm.Nt):
        Fx_t = np.cumsum(hm.px_smooth[tt, :])
        q_ref[:, 0, tt] = np.interp(q, Fx_t, x_grid)

    assert np.allclose(q_vals, q_ref, atol=1e-10)


def test_sample_and_sampled_vals_shapes_and_support():
    hm, x_grid = _build_hm()
    hm.sample(40)
    assert hm.ix_sample.shape == (hm.Nt, 40)
    assert hm.ix_sample.dtype.kind in {"i", "u"}
    assert hm.ix_sample.min() >= 0
    assert hm.ix_sample.max() < hm.Nx

    vals = hm.sampled_vals(x_grid)
    assert vals.shape == (hm.Nt, 1, 40)
