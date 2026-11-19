r"""Chinchilla "Approach 3" parametric law, an LR-schedule-aware extension, and the
compute-optimal frontier.

    L(N, D)    = E + A N^{-alpha} + B D^{-beta}                        (Hoffmann et al. 2022)
    L(N, D, f) = E + A N^{-alpha} + B D^{-beta} + K [h(f) - h(1)]       (schedule-aware)

``h(f) = eta(f) / eta_max`` is the learning-rate multiplier at fraction ``f`` of the
run. Intermediate checkpoints of a decaying schedule sit above the loss a run
*annealed at that D* would reach; to first order the excess scales with the current
LR (SGD noise-floor argument; cf. Tissue et al. 2024's annealing-area term). The
term vanishes at f = 1, so (E, A, B, alpha, beta) describe annealed end-of-run losses,
which is the surface the compute-optimal question is about.

Fitting follows Hoffmann et al. / Besiroglu et al. (2024): theta = (a, b, e, alpha, beta)
with A=e^a, B=e^b, E=e^e, prediction LSE(a - alpha log N, b - beta log D, e) (+ K h),
Huber loss on log residuals, L-BFGS-B from the best points of a 4,500-point grid.
``mse_log`` / ``mse_lin`` objectives are exposed because the loss function is one of
the audited choices.

Compute-optimal allocation under C = 6 N D:
    N_opt = G (C/6)^{beta/(alpha+beta)},  D_opt = G^{-1} (C/6)^{alpha/(alpha+beta)},
    G = (alpha A / (beta B))^{1/(alpha+beta)},
    r(C) = D_opt / N_opt = G^{-2} (C/6)^{(alpha-beta)/(alpha+beta)}.
Unless alpha == beta, r depends on C: a tokens-per-parameter number is only
meaningful at a stated compute budget.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

CHINCHILLA_C = 5.76e23  # Chinchilla-70B training compute
NAMES = ("a", "b", "e", "alpha", "beta")
BOUNDS = [(-20.0, 60.0), (-20.0, 60.0), (-10.0, 3.0), (1e-3, 4.0), (1e-3, 4.0)]
K_BOUNDS = (0.0, 50.0)


@dataclass
class FitConfig:
    loss: str = "huber"  # huber | mse_log | mse_lin
    delta: float = 1e-3  # Huber delta (Hoffmann et al. value)
    tie_exponents: bool = False  # constrain alpha == beta
    schedule: bool = False  # add the K [h(f) - h(1)] term (needs h)
    n_starts: int = 32  # best grid points refined with L-BFGS


@dataclass
class Fit:
    theta: np.ndarray  # (a, b, e, alpha, beta)
    objective: float
    converged: bool
    at_bound: bool
    n: int
    K: float = 0.0
    cfg: FitConfig = field(repr=False, default_factory=FitConfig)

    A = property(lambda s: float(np.exp(s.theta[0])))
    B = property(lambda s: float(np.exp(s.theta[1])))
    E = property(lambda s: float(np.exp(s.theta[2])))
    alpha = property(lambda s: float(s.theta[3]))
    beta = property(lambda s: float(s.theta[4]))

    def predict(self, N, D, h=None):
        return predict(self.theta, N, D, h, self.K)

    def tokens_per_param(self, C=CHINCHILLA_C):
        return tokens_per_param(self.theta, C)

    def as_dict(self) -> dict:
        return {"E": self.E, "A": self.A, "B": self.B, "alpha": self.alpha, "beta": self.beta,
                "K": self.K, "objective": self.objective, "converged": self.converged,
                "at_bound": self.at_bound, "n": self.n}


# ----------------------------------------------------------------------------- schedules
def lr_multiplier(f, warmup=0.01, final=0.1):
    """Linear warmup then cosine decay to ``final`` x peak (GPT-NeoX / OLMo style)."""
    f = np.asarray(f, float)
    cos = final + (1 - final) * 0.5 * (1 + np.cos(np.pi * np.clip((f - warmup) / (1 - warmup), 0, 1)))
    return np.where(f < warmup, f / warmup, cos)


# ----------------------------------------------------------------------------- model
def _terms(theta, logN, logD):
    a, b, e, al, be = theta[:5]
    return np.stack([a - al * logN, b - be * logD, np.full_like(logN, e)])


def predict(theta, N, D, h=None, K=0.0):
    base = np.exp(logsumexp(_terms(np.asarray(theta, float), np.log(N), np.log(D)), axis=0))
    return base if h is None else base + K * np.asarray(h)


def _rho(r, cfg: FitConfig):
    """Return (loss, d loss / d r)."""
    if cfg.loss == "huber":
        d = cfg.delta
        ar = np.abs(r)
        return np.where(ar <= d, 0.5 * r * r, d * (ar - 0.5 * d)), np.clip(r, -d, d)
    return 0.5 * r * r, r


def objective_and_grad(theta, logN, logD, logL, cfg: FitConfig, w=None, h=None):
    """theta = 5 law params (+ K if h is given). Returns (objective, gradient)."""
    T = _terms(theta, logN, logD)
    lse = logsumexp(T, axis=0)
    W = np.exp(T - lse)
    base = np.exp(lse)
    pred = base if h is None else base + theta[5] * h
    if cfg.loss == "mse_lin":
        val, dr = _rho(np.exp(logL) - pred, cfg)
        dpred = -dr  # d val / d pred
    else:
        val, dr = _rho(logL - np.log(pred), cfg)
        dpred = -dr / pred
    if w is not None:
        val, dpred = val * w, dpred * w
    dlse = dpred * base  # chain through base = exp(lse)
    g = [(dlse * W[0]).sum(), (dlse * W[1]).sum(), (dlse * W[2]).sum(),
         (dlse * W[0] * -logN).sum(), (dlse * W[1] * -logD).sum()]
    if h is not None:
        g.append((dpred * h).sum())
    return val.sum(), np.array(g)


# ----------------------------------------------------------------------------- fitting
def init_grid(tie=False):
    """Hoffmann et al.-style grid (their App. D.2): 4,500 points (900 if tied)."""
    al = np.arange(0.0, 2.01, 0.5)
    es = np.arange(-1.0, 1.01, 0.5)
    ab = np.arange(0.0, 25.1, 5.0)
    if tie:
        g = [(a, b, e, x, x) for a, b, e, x in itertools.product(ab, ab, es, al)]
    else:
        g = list(itertools.product(ab, ab, es, al, al))
    g = np.array(g, float)
    g[:, 3:] = np.maximum(g[:, 3:], 1e-3)
    return g


def grid_objective(grid, logN, logD, logL, cfg: FitConfig, w=None, chunk=500):
    """Vectorised objective (K = 0) over rows of ``grid``."""
    out = np.empty(len(grid))
    for i in range(0, len(grid), chunk):
        g = grid[i:i + chunk]
        T = np.stack([g[:, [0]] - g[:, [3]] * logN, g[:, [1]] - g[:, [4]] * logD,
                      np.broadcast_to(g[:, [2]], (len(g), logN.size))])
        lse = logsumexp(T, axis=0)
        r = np.exp(logL) - np.exp(lse) if cfg.loss == "mse_lin" else logL - lse
        v, _ = _rho(r, cfg)
        out[i:i + chunk] = (v * w).sum(1) if w is not None else v.sum(1)
    return out


def _scale(cfg: FitConfig, n: int) -> float:
    # Rescale so L-BFGS tolerances are meaningful for tiny Huber objectives.
    return 1.0 / (n * (cfg.delta ** 2 if cfg.loss == "huber" else 1e-4))


def _free_to_full(x, cfg):
    """Free vector -> (5 law params, K)."""
    if cfg.tie_exponents:
        th = np.r_[x[:4], x[3]]
        rest = x[4:]
    else:
        th, rest = x[:5], x[5:]
    return th, (rest[0] if cfg.schedule else 0.0)


def _full_to_free(th, K, cfg):
    th = np.asarray(th, float)
    x = np.r_[th[:3], th[3:5].mean()] if cfg.tie_exponents else th[:5]
    return np.r_[x, K] if cfg.schedule else x


def fit(N, D, L, cfg: FitConfig | None = None, starts=None, weights=None, h=None,
        start_K=0.0) -> Fit:
    cfg = cfg or FitConfig()
    if cfg.schedule and h is None:
        raise ValueError("schedule=True needs h")
    h = np.asarray(h, float) if cfg.schedule else None
    logN, logD, logL = np.log(N), np.log(D), np.log(L)
    n = len(logL)
    s = _scale(cfg, n)
    w = None if weights is None else np.asarray(weights, float) / np.mean(weights)

    def f(x):
        th, K = _free_to_full(x, cfg)
        v, g = objective_and_grad(np.r_[th, K] if cfg.schedule else th, logN, logD, logL, cfg, w, h)
        gl = np.r_[g[:3], g[3] + g[4]] if cfg.tie_exponents else g[:5]
        if cfg.schedule:
            gl = np.r_[gl, g[5]]
        return v * s, gl * s

    bounds = (BOUNDS[:4] if cfg.tie_exponents else BOUNDS) + ([K_BOUNDS] if cfg.schedule else [])

    # Starts = caller's warm starts (e.g. the full-data optimum inside a bootstrap)
    # + the n_starts best Hoffmann-grid points under *this* data (K = 0 and K = start_K').
    pool = []
    if starts is not None:
        for st in np.atleast_2d(np.asarray(starts, float)):
            pool.append(_full_to_free(st[:5], st[5] if len(st) > 5 else start_K, cfg))
    if cfg.n_starts > 0:
        grid = init_grid(cfg.tie_exponents)
        top = grid[np.argsort(grid_objective(grid, logN, logD, logL, cfg, w))[: cfg.n_starts]]
        for st in top:
            pool.append(_full_to_free(st, 0.0, cfg))
            if cfg.schedule:
                pool.append(_full_to_free(st, 0.05, cfg))

    best = None
    for x0 in pool:
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    th, K = _free_to_full(best.x, cfg)
    at_bound = any(abs(th[i] - lo) < 1e-6 or abs(th[i] - hi) < 1e-6
                   for i, (lo, hi) in enumerate(BOUNDS))
    # L-BFGS-B often reports ABNORMAL_TERMINATION in the line search at a genuine
    # optimum of a piecewise-quadratic objective; treat a tiny projected gradient as converged.
    converged = bool(best.success or np.max(np.abs(best.jac)) < 1e-5 * max(1.0, abs(best.fun)))
    return Fit(np.asarray(th, float), float(best.fun / s), converged, bool(at_bound), n,
               float(K), cfg)


# ----------------------------------------------------------------------------- optimum
def optimal_allocation(theta, C):
    a, b, e, al, be = np.asarray(theta, float)[:5]
    A, B = np.exp(a), np.exp(b)
    G = (al * A / (be * B)) ** (1.0 / (al + be))
    C6 = np.asarray(C, float) / 6.0
    return G * C6 ** (be / (al + be)), C6 ** (al / (al + be)) / G


def tokens_per_param(theta, C=CHINCHILLA_C):
    N, D = optimal_allocation(theta, C)
    return D / N


def compute_for_ratio_check(theta, C):
    """Numerical argmin of L(N, C/6N) over log N (used by tests)."""
    from scipy.optimize import minimize_scalar
    th = np.asarray(theta, float)
    r = minimize_scalar(lambda lN: predict(th, np.exp(lN), C / (6 * np.exp(lN))),
                        bounds=(np.log(1e3), np.log(C / 6)), method="bounded",
                        options={"xatol": 1e-10})
    N = np.exp(r.x)
    return N, C / (6 * N)
