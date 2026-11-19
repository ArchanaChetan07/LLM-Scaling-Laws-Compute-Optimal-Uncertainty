import numpy as np
import pytest
from scipy.optimize import approx_fprime

from scalingci import law

TRUE = np.array([np.log(400.0), np.log(1800.0), np.log(1.8), 0.34, 0.28])


def synth(theta=TRUE, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    N = np.repeat(np.geomspace(5e7, 1e10, 8), 20)
    D = np.tile(np.geomspace(2e9, 5e11, 20), 8)
    L = law.predict(theta, N, D) * np.exp(noise * rng.standard_normal(N.size))
    return N, D, L


@pytest.mark.parametrize("loss", ["huber", "mse_log", "mse_lin"])
def test_gradient_matches_finite_difference(loss):
    N, D, L = synth(noise=0.02)
    cfg = law.FitConfig(loss=loss, delta=1e-2)
    th = TRUE + np.array([0.3, -0.2, 0.05, 0.02, -0.03])
    f = lambda t: law.objective_and_grad(t, np.log(N), np.log(D), np.log(L), cfg)[0]
    g = law.objective_and_grad(th, np.log(N), np.log(D), np.log(L), cfg)[1]
    np.testing.assert_allclose(g, approx_fprime(th, f, 1e-7), rtol=2e-3, atol=1e-6)


def test_noiseless_recovery():
    N, D, L = synth()
    fit = law.fit(N, D, L)
    np.testing.assert_allclose(fit.theta, TRUE, atol=2e-3)
    assert fit.tokens_per_param() == pytest.approx(law.tokens_per_param(TRUE), rel=1e-2)


def test_closed_form_optimum_matches_numerical():
    for C in [1e20, 5.76e23, 1e26]:
        N, D = law.optimal_allocation(TRUE, C)
        Nn, Dn = law.compute_for_ratio_check(TRUE, C)
        assert N == pytest.approx(Nn, rel=1e-4)
        assert D * N * 6 == pytest.approx(C, rel=1e-9)


def test_ratio_constant_when_exponents_equal():
    th = TRUE.copy(); th[4] = th[3]
    r = [law.tokens_per_param(th, C) for C in (1e19, 1e23, 1e27)]
    np.testing.assert_allclose(r, r[0], rtol=1e-10)


def test_tied_fit_has_equal_exponents():
    N, D, L = synth(noise=0.01)
    fit = law.fit(N, D, L, law.FitConfig(tie_exponents=True, n_starts=8))
    assert fit.alpha == fit.beta


def _sched_synth(K=0.15, noise=0.0):
    N, D, _ = synth()
    f = np.tile(np.geomspace(0.05, 1.0, 20), 8)
    h = law.lr_multiplier(f, 0.01, 0.1) - 0.1
    L = law.predict(TRUE, N, D, h, K) * np.exp(noise * np.random.default_rng(1).standard_normal(N.size))
    return N, D, L, h


def test_schedule_gradient():
    N, D, L, h = _sched_synth(noise=0.01)
    cfg = law.FitConfig(delta=1e-2, schedule=True)
    th = np.r_[TRUE + 0.05, 0.1]
    fn = lambda t: law.objective_and_grad(t, np.log(N), np.log(D), np.log(L), cfg, h=h)[0]
    g = law.objective_and_grad(th, np.log(N), np.log(D), np.log(L), cfg, h=h)[1]
    np.testing.assert_allclose(g, approx_fprime(th, fn, 1e-7), rtol=2e-3, atol=1e-6)


def test_schedule_recovery_and_naive_bias():
    N, D, L, h = _sched_synth()
    good = law.fit(N, D, L, law.FitConfig(schedule=True), h=h)
    assert good.K == pytest.approx(0.15, rel=1e-2)
    np.testing.assert_allclose(good.theta, TRUE, atol=5e-3)
    naive = law.fit(N, D, L)  # ignores the schedule -> biased ratio
    r_true = law.tokens_per_param(TRUE)
    assert abs(np.log(naive.tokens_per_param() / r_true)) > 5 * abs(np.log(good.tokens_per_param() / r_true))
