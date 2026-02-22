import numpy as np
import pytest

from py_tools.bayesian.mcmc import (
    MonteCarlo,
    RWMC,
    SMC,
    adapt_jump_scale,
    check_bounds,
    importance_sample,
    metropolis_step,
    numerical_to_bool_blocks,
    randomize_blocks,
    rwmh,
)
from py_tools.bayesian.prior import Prior


def test_adapt_jump_scale_bounds():
    lo = adapt_jump_scale(0.0, adapt_sens=16.0, adapt_target=0.25, adapt_range=0.1)
    hi = adapt_jump_scale(1.0, adapt_sens=16.0, adapt_target=0.25, adapt_range=0.1)
    assert 0.95 <= lo <= 1.05
    assert 0.95 <= hi <= 1.05
    assert hi > lo


def test_block_helpers_cover_all_indices():
    blocks = randomize_blocks(nx=7, nblock=3)
    assert len(blocks) == 3
    covered = np.zeros(7, dtype=int)
    for block in blocks:
        covered += block.astype(int)
    assert np.all(covered == 1)

    bool_blocks = numerical_to_bool_blocks([np.array([0, 2]), np.array([1, 3])], nx=4)
    assert len(bool_blocks) == 2
    assert np.array_equal(bool_blocks[0], np.array([True, False, True, False]))


def test_check_bounds():
    x = np.array([0.0, 1.0])
    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 2.0])
    assert check_bounds(x, lb, ub)
    assert not check_bounds(np.array([2.0, 1.0]), lb, ub)


def test_metropolis_step_accept_and_reject():
    f = lambda z: -np.sum(z * z)
    x = np.array([0.0])
    x_try = np.array([1.0])
    # Force reject
    x_new, post_new, acc = metropolis_step(f, x, x_try, post=f(x), log_u=0.0)
    assert np.array_equal(x_new, x)
    assert acc is False
    # Force accept with very negative log_u
    x_new, post_new, acc = metropolis_step(f, x, x_try, post=f(x), log_u=-10.0)
    assert np.array_equal(x_new, x_try)
    assert acc is True


def test_importance_sample_serial_shapes():
    from scipy.stats import multivariate_normal as mv

    f = lambda z: -0.5 * np.sum(z * z)
    dist = mv(mean=np.zeros(2), cov=np.eye(2))

    draws, lw = importance_sample(f, dist, Nsim=1, Nx=2, parallel=False)
    assert draws.shape == (1, 2)
    assert lw.shape == (1,)

    draws, lw = importance_sample(f, dist, Nsim=5, Nx=2, parallel=False)
    assert draws.shape == (5, 2)
    assert lw.shape == (5,)


def test_rwmh_default_blocks_smoke():
    post = lambda z: -0.5 * np.sum(z * z)
    x_store, p_store, acc = rwmh(post, np.array([0.0]), Nstep=4)
    assert x_store.shape == (4, 1)
    assert p_store.shape == (4,)
    assert 0.0 <= acc <= 1.0


def test_montecarlo_bounds_and_iterative_find_mode_without_names():
    prior = Prior()
    prior.add("norm", mean=0.0, sd=1.0)
    mc = MonteCarlo(log_like=lambda x: -0.5 * np.sum(x * x), prior=prior, ub=np.array([1.0]))
    assert np.isneginf(mc.lb[0])
    assert mc.ub[0] == 1.0
    assert np.isfinite(mc.posterior(np.array([0.0])))

    # names=None should still work with iterate=True
    res = mc.find_mode(np.array([0.2]), iterate=True, disp_iterate=True, method="Nelder-Mead")
    assert np.isfinite(mc.post_mode)
    assert res.success or np.isfinite(res.fun)


def test_montecarlo_importance_sample_method():
    prior = Prior()
    prior.add("norm", mean=0.0, sd=1.0)
    mc = MonteCarlo(log_like=lambda x: -0.5 * np.sum(x * x), prior=prior, Nx=1)
    mc.x_mode = np.array([0.0])
    mc.H_inv = np.array([[1.0]])
    draws, lw, ess = mc.importance_sample(10, parallel=False)
    assert draws.shape == (10, 1)
    assert lw.shape == (10,)
    assert np.isfinite(ess)


def test_rwmc_sample_guard_and_run_all(tmp_path):
    prior = Prior()
    prior.add("norm", mean=0.0, sd=1.0, name="theta")

    rw = RWMC(log_like=lambda x: -0.5 * np.sum(x * x), prior=prior, Nx=1, out_dir=None, suffix=None)
    rw.initialize(x0=np.array([0.0]), C=np.eye(1), stride=1)
    with pytest.raises(ValueError):
        rw.sample(Nsim=4, n_save=2, log=False)

    rw2 = RWMC(
        log_like=lambda x: -0.5 * np.sum(x * x),
        prior=prior,
        Nx=1,
        out_dir=str(tmp_path),
        suffix="chain",
    )
    rw2.run_all(
        np.array([0.1]),
        Nsim=5,
        mode_kwargs={"method": "Nelder-Mead"},
        init_kwargs={"stride": 1},
        sample_kwargs={"log": False},
    )
    assert rw2.draws.shape == (5, 1)
    assert rw2.post_sim.shape == (5,)


def test_smc_initialize_guard_and_serial_sample(tmp_path):
    prior = Prior()
    prior.add("norm", mean=0.0, sd=1.0)

    smc_bad = SMC(log_like=lambda x: -0.5 * np.sum(x * x), prior=prior, Nx=1, out_dir=None)
    with pytest.raises(ValueError):
        smc_bad.initialize(Npt=8, Nstep=3, parallel=False, save_intermediate=True)

    smc = SMC(log_like=lambda x: -0.5 * np.sum(x * x), prior=prior, Nx=1, out_dir=str(tmp_path))
    smc.initialize(Npt=8, Nstep=3, parallel=False, save_intermediate=False)
    smc.sample(quiet=True)
    assert smc.draws.shape == (3, 8, 1)
    assert smc.post.shape == (3, 8)
    assert np.all(np.isfinite(smc.C_star))
