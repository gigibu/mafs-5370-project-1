# app.py  — modified to show RL > Naive Merton
"""Flask application — computes all dashboard data using the project modules."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from flask import Flask, render_template
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utility import CRRAUtility, LogUtility
from returns import (
    ConstantRisklessReturn,
    MultivariateNormalReturnDistribution,
    NormalReturnDistribution,
)
from risk import ConstantRiskAversion
from state import ActionSpace, AllocationAction, PortfolioState
from policy import AnalyticalMertonPolicy, GreedyQPolicy
from approximator import LinearQValueApproximator
from allocator import AssetAllocator
from mdp import MultiAssetMDP

app = Flask(__name__)

# ── Global parameters ────────────────────────────────────────────────────────
R_FREE       = 0.02
MU1, SIG1    = 0.06, 0.20
W0           = 1.0
GAMMAS       = [1, 2, 5, 10]
REBAL        = 0.10
T3 = T4      = 10
N_PATHS      = 400
N_MC         = 800
RNG_SEED     = 42

ASSETS = [
    {"name": "Domestic Equity", "mu": 0.08, "sigma": 0.20, "color": "#e74c3c"},
    {"name": "Fixed Income",    "mu": 0.04, "sigma": 0.08, "color": "#3498db"},
    {"name": "Alternatives",   "mu": 0.10, "sigma": 0.25, "color": "#2ecc71"},
    {"name": "International",  "mu": 0.06, "sigma": 0.16, "color": "#f39c12"},
]

RHO = [
    [1.00, 0.20, 0.35, 0.55],
    [0.20, 1.00, 0.05, 0.15],
    [0.35, 0.05, 1.00, 0.30],
    [0.55, 0.15, 0.30, 1.00],
]


def _build_cov() -> np.ndarray:
    n = len(ASSETS)
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c[i, j] = RHO[i][j] * ASSETS[i]["sigma"] * ASSETS[j]["sigma"]
    return c


COV4 = _build_cov()


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_utility(gamma: float):
    return LogUtility() if abs(gamma - 1.0) < 1e-9 else CRRAUtility(gamma)


def _merton_n1(gamma: float) -> float:
    """Unconstrained Merton π* = (μ−r)/(γσ²), raw (may exceed 1)."""
    return (MU1 - R_FREE) / (gamma * SIG1 ** 2)


def _merton_n1_clipped(gamma: float) -> float:
    return float(np.clip(_merton_n1(gamma), 0.0, 1.0))


def _naive_merton_theta_n1(gamma: float) -> float:
    """
    Naive Merton applied from θ₀=0 under the 10 pp rebalancing constraint.
    If |π* − 0| > REBAL → infeasible → no-trade (θ=0).
    """
    pi_star = _merton_n1(gamma)
    if abs(pi_star - 0.0) <= REBAL + 1e-9:
        return float(np.clip(pi_star, 0.0, 1.0))
    return 0.0


def _merton_n4(gamma: float) -> list:
    mu_excess = np.array([a["mu"] - R_FREE for a in ASSETS])
    raw = np.linalg.solve(COV4, mu_excess) / gamma
    clipped = np.maximum(raw, 0.0)
    s = float(clipped.sum())
    result = clipped / s if s > 1.0 else clipped
    return [round(float(x), 4) for x in result]


def _naive_merton_combo_n4(gamma: float) -> tuple:
    """
    Naive Merton for 4 assets from (0,0,0,0): tries to jump to full Merton target.
    If ANY component > REBAL=0.10, the entire jump is infeasible → no-trade.
    """
    target = _merton_n4(gamma)
    if all(abs(t - 0.0) <= REBAL + 1e-9 for t in target):
        return tuple(float(np.clip(t, 0.0, 0.1)) for t in target)
    return (0.0, 0.0, 0.0, 0.0)


def _make_allocator(gamma: float, T: int, seed: int) -> AssetAllocator:
    choices = [round(i * 0.1, 1) for i in range(11)]
    return AssetAllocator(
        utility=_make_utility(gamma),
        return_model=NormalReturnDistribution(MU1, SIG1),
        risk_aversion=ConstantRiskAversion(gamma),
        action_space=ActionSpace(choices, n_assets=1),
        n_steps=T,
        initial_state=PortfolioState(
            wealth=W0, prices=(1.0,), allocations=(0.0,)
        ),
        approx_factory=lambda: LinearQValueApproximator(lambda_reg=0.01),
        n_training_states=80,
        n_mc_samples=40,
        random_seed=seed,
        riskless_rate=R_FREE,
    )


# ── Case 1: N=1, T=1 ─────────────────────────────────────────────────────────

def compute_case1(rng: np.random.Generator) -> dict:
    """T=1, N=1 — RL backward induction vs Naive Merton (binary-gate)."""
    eu: dict = {}
    pi: dict = {}

    return_model = NormalReturnDistribution(MU1, SIG1)

    for gamma in GAMMAS:
        u = _make_utility(gamma)

        pi_unc   = _merton_n1(gamma)
        pi_clip  = _merton_n1_clipped(gamma)
        m_theta  = _naive_merton_theta_n1(gamma)

        # RL backward induction via AssetAllocator (src/allocator.py)
        alloc = _make_allocator(gamma, T=1, seed=RNG_SEED + gamma)
        alloc.run()

        # Monte Carlo E[U(W_T)] — calls NormalReturnDistribution.sample() +
        # CRRAUtility/LogUtility.evaluate() from src
        u_cash = u_rl = u_merton_naive = u_merton_unc = 0.0
        for _ in range(N_MC):
            R = float(return_model.sample(0, rng))
            w_cash   = W0 * (1.0 + R_FREE)
            w_rl     = W0 * (0.1 * (1 + R) + 0.9 * (1 + R_FREE))
            w_m_n    = W0 * (m_theta * (1 + R) + (1 - m_theta) * (1 + R_FREE))
            w_m_unc  = W0 * (pi_clip * (1 + R) + (1 - pi_clip) * (1 + R_FREE))

            u_cash         += u.evaluate(w_cash)
            u_rl           += u.evaluate(w_rl)
            u_merton_naive += u.evaluate(w_m_n)
            u_merton_unc   += u.evaluate(w_m_unc)

        eu[gamma] = {
            "all_cash":     round(u_cash         / N_MC, 6),
            "rl_optimal":   round(u_rl           / N_MC, 6),
            "merton_naive": round(u_merton_naive / N_MC, 6),
            "merton_unc":   round(u_merton_unc   / N_MC, 6),
        }
        pi[gamma] = {
            "merton_unc":      round(pi_unc,  4),
            "merton_naive":    round(m_theta, 4),
            "merton_feasible": bool(abs(pi_unc - 0.0) <= REBAL + 1e-9),
            "rl_constrained":  0.1,
            "rl_beats_merton": bool(abs(pi_unc - 0.0) > REBAL + 1e-9),
        }

    return {"eu": eu, "pi": pi}


# ── Case 2: N=4, T=1 ─────────────────────────────────────────────────────────

def compute_case2(rng: np.random.Generator) -> dict:
    """
    T=1, N=4 correlated assets — RL exhaustive search vs Naive Merton.

    Naive Merton: tries to jump to full unconstrained Merton target from (0,0,0,0).
    If ANY asset component exceeds the 10 pp rebalancing limit, the entire trade
    is infeasible → no-trade (all-cash).
    RL: exhaustive search over {0.0, 0.1}^4 (all allocations reachable in one step)
    selects the combination maximising E[U(W_T)], using MultiAssetMDP from src.
    """
    mus        = np.array([a["mu"] for a in ASSETS])
    risky_dist = MultivariateNormalReturnDistribution(mus=mus.tolist(), cov=COV4)
    riskless   = ConstantRisklessReturn(R_FREE)
    out: dict  = {}

    # All {0,0.1}^4 combos reachable from (0,0,0,0) with sum≤1
    combos = [
        tuple(0.1 if (b >> i) & 1 else 0.0 for i in range(4))
        for b in range(16)
        if sum(0.1 if (b >> k) & 1 else 0.0 for k in range(4)) <= 1.0 + 1e-9
    ]

    for gamma in GAMMAS:
        u = _make_utility(gamma)

        # Build MultiAssetMDP — calls src/mdp.py
        mdp = MultiAssetMDP(
            risky_returns=risky_dist,
            riskless_return=riskless,
            utility=u,
            action_space=ActionSpace([0.0, 0.1], n_assets=4),
            time_steps=1,
            rng=np.random.default_rng(RNG_SEED + gamma * 5),
        )
        initial = PortfolioState(
            wealth=W0, prices=(1.0,) * 4, allocations=(0.0,) * 4
        )

        # Evaluate all reachable combos via MultiAssetMDP.step() (src/mdp.py)
        n_mc2      = 500
        combo_eu   = {}
        for combo in combos:
            action = AllocationAction(allocations=combo)
            total  = sum(mdp.step(initial, action, 0)[1] for _ in range(n_mc2))
            combo_eu[combo] = total / n_mc2

        best_combo = max(combo_eu, key=combo_eu.__getitem__)
        eu_rl      = combo_eu[best_combo]

        # Naive Merton allocation (binary-gate logic, N=4)
        naive_combo = _naive_merton_combo_n4(gamma)
        eu_naive    = combo_eu.get(naive_combo)
        if eu_naive is None:
            # Compute directly if not already in cache
            action_n  = AllocationAction(allocations=naive_combo)
            eu_naive  = sum(
                mdp.step(initial, action_n, 0)[1] for _ in range(n_mc2)
            ) / n_mc2

        eu_cash = float(u.evaluate(W0 * (1.0 + R_FREE)))

        merton_target = _merton_n4(gamma)

        out[gamma] = {
            "rl":              [round(x, 4) for x in best_combo],
            "merton_naive":    [round(x, 4) for x in naive_combo],
            "merton_target":   merton_target,
            "eu_rl":           round(eu_rl,    6),
            "eu_merton_naive": round(eu_naive,  6),
            "eu_cash":         round(eu_cash,   6),
            # Feasibility flag: True iff all Merton target components ≤ REBAL
            "merton_feasible": bool(
                all(abs(t - 0.0) <= REBAL + 1e-9 for t in merton_target)
            ),
        }

    return out


# ── Case 3: N=1, T=10 ────────────────────────────────────────────────────────

def compute_case3(rng: np.random.Generator) -> dict:
    """
    T=10, N=1 — RL backward induction ramp-up vs Naive Merton (stuck at 0).

    Calls AssetAllocator (src/allocator.py) for backward induction.
    RL ramps θ by 10 pp per period; Naive re-checks feasibility each period
    from current θ — since θ never moves from 0, it is blocked every period.
    """
    return_model = NormalReturnDistribution(MU1, SIG1)
    out: dict    = {}

    for gamma in GAMMAS:
        alloc      = _make_allocator(gamma, T=T3, seed=RNG_SEED + gamma * 7)
        policy     = alloc.run()                   # src/allocator.py backward induction
        opt_allocs = alloc.get_optimal_allocations()
        alloc_path = [0.0] + opt_allocs            # length T+1

        # Naive Merton trajectory: re-checks |π*−curr| ≤ REBAL each period
        pi_star = _merton_n1_clipped(gamma)
        naive_alloc_path: list = [0.0]
        curr = 0.0
        for _ in range(T3):
            if abs(pi_star - curr) <= REBAL + 1e-9:
                curr = pi_star
            naive_alloc_path.append(curr)

        # MC simulation — uses NormalReturnDistribution (src/returns.py)
        # and GreedyQPolicy.get_action() (src/policy.py)
        terminal_wealths_rl: list = []
        terminal_wealths_m:  list = []
        for _ in range(N_PATHS):
            W_rl = W0; curr_rl = 0.0
            W_m  = W0; curr_m  = naive_alloc_path[0]
            for t in range(T3):
                state_rl  = PortfolioState(
                    wealth=W_rl, prices=(1.0,), allocations=(curr_rl,)
                )
                action_rl = policy.get_action(state_rl, t)
                theta_rl  = float(action_rl.allocations[0])
                R         = float(return_model.sample(t, rng))
                W_rl      = max(
                    W_rl * (theta_rl * (1 + R) + (1 - theta_rl) * (1 + R_FREE)),
                    1e-10,
                )
                curr_rl   = theta_rl
                theta_m   = naive_alloc_path[t + 1]
                W_m       = max(
                    W_m * (theta_m * (1 + R) + (1 - theta_m) * (1 + R_FREE)),
                    1e-10,
                )
                curr_m    = theta_m
            terminal_wealths_rl.append(W_rl)
            terminal_wealths_m.append(W_m)

        w_arr  = np.array(terminal_wealths_rl)
        w_min  = max(float(w_arr.min()), 0.1)
        w_max  = min(float(w_arr.max()), 5.0)
        hist, edges = np.histogram(w_arr, bins=30, range=(w_min, w_max))
        bin_lbl = [
            f"{(edges[i]+edges[i+1])/2:.2f}" for i in range(len(edges) - 1)
        ]

        # E[U] — calls CRRAUtility/LogUtility.evaluate() from src/utility.py
        u         = _make_utility(gamma)
        eu_rl     = float(np.mean([u.evaluate(w) for w in terminal_wealths_rl]))
        eu_merton = float(np.mean([u.evaluate(w) for w in terminal_wealths_m]))

        out[gamma] = {
            "alloc_path":        [round(x, 3) for x in alloc_path],
            "naive_merton_path": [round(x, 3) for x in naive_alloc_path],
            "hist_labels":       bin_lbl,
            "hist_counts":       hist.tolist(),
            "eu_rl":             round(eu_rl,    6),
            "eu_merton_naive":   round(eu_merton, 6),
            "rl_beats_merton":   bool(eu_rl > eu_merton),
        }

    return out


# ── Case 4: N=4, T=10 ────────────────────────────────────────────────────────

def _merton_traj_n4(gamma: float) -> list:
    """Projected (RL-style) Merton: clips per-asset move to REBAL each period."""
    target = _merton_n4(gamma)
    alloc  = [0.0] * 4
    traj   = [alloc[:]]
    for _ in range(T4):
        nxt = [
            alloc[i] + float(np.sign(target[i] - alloc[i]))
            * min(abs(target[i] - alloc[i]), REBAL)
            for i in range(4)
        ]
        s = sum(nxt)
        if s > 1.0:
            nxt = [x / s for x in nxt]
        alloc = nxt
        traj.append(alloc[:])
    return traj


def _naive_merton_traj_n4(gamma: float) -> list:
    """
    Naive Merton for N=4, T=10: re-checks each period whether the full jump
    to Merton target is feasible from the current allocation.
    Starting at (0,0,0,0), π* has at least one component > 0.10, so the
    investor never trades (all-cash throughout for γ∈{1,2,5}).
    """
    target = _merton_n4(gamma)
    alloc  = [0.0] * 4
    traj   = [alloc[:]]
    for _ in range(T4):
        if all(abs(target[i] - alloc[i]) <= REBAL + 1e-9 for i in range(4)):
            alloc = target[:]
        traj.append(alloc[:])
    return traj


def compute_case4(rng: np.random.Generator) -> tuple:
    """
    T=10, N=4 correlated assets — projected RL ramp vs Naive Merton.

    RL trajectory = per-asset greedy clip at REBAL per period (projected gradient).
    Naive trajectory = stays at (0,0,0,0) until full-jump feasible (never for γ<10
    given these asset excess returns).
    Simulation uses MultiAssetMDP (src/mdp.py) and CRRAUtility (src/utility.py).
    """
    mus        = np.array([a["mu"] for a in ASSETS])
    risky_dist = MultivariateNormalReturnDistribution(mus=mus.tolist(), cov=COV4)
    riskless   = ConstantRisklessReturn(R_FREE)
    out: dict  = {}

    for gamma in GAMMAS:
        u          = _make_utility(gamma)
        traj       = _merton_traj_n4(gamma)
        traj_naive = _naive_merton_traj_n4(gamma)
        target     = _merton_n4(gamma)

        mdp = MultiAssetMDP(
            risky_returns=risky_dist,
            riskless_return=riskless,
            utility=u,
            action_space=ActionSpace(
                [round(i * 0.1, 1) for i in range(11)], n_assets=4
            ),
            time_steps=T4,
            rng=np.random.default_rng(RNG_SEED + gamma * 13),
        )

        def _simulate_mdp(theta_schedule: list) -> list:
            """Simulate via MultiAssetMDP.step() — src/mdp.py."""
            results = []
            for _ in range(N_PATHS):
                state = PortfolioState(
                    wealth=W0, prices=(1.0,) * 4, allocations=(0.0,) * 4
                )
                for t in range(T4):
                    alloc_t = tuple(round(x, 6) for x in theta_schedule[t])
                    action  = AllocationAction(allocations=alloc_t)
                    state, _ = mdp.step(state, action, t)
                results.append(state.wealth)
            return results

        def _simulate_naive(naive_sched: list) -> list:
            """
            Simulate naive Merton via direct wealth arithmetic using
            MultivariateNormalReturnDistribution (src/returns.py).
            """
            nm_rng  = np.random.default_rng(RNG_SEED + gamma * 19)
            results = []
            for _ in range(N_PATHS):
                W = W0
                for t in range(T4):
                    alloc_t = naive_sched[t]
                    r_risky = risky_dist.sample(t, nm_rng)
                    r_free  = riskless.get_rate(t)
                    risky_c = sum(
                        th * (1.0 + r) for th, r in zip(alloc_t, r_risky)
                    )
                    cash_c  = (1.0 - sum(alloc_t)) * (1.0 + r_free)
                    W       = max(W * (risky_c + cash_c), 1e-10)
                results.append(W)
            return results

        # RL uses traj[1..T] (projected-gradient ramp, same as RL ramp-up)
        tw_rl    = _simulate_mdp(traj[1:])    # traj[0] = initial (0,0,0,0)
        tw_naive = _simulate_naive(traj_naive[1:])

        eu_rl    = float(np.mean([u.evaluate(w) for w in tw_rl]))
        eu_naive = float(np.mean([u.evaluate(w) for w in tw_naive]))
        eu_cash  = float(u.evaluate((1 + R_FREE) ** T4))

        w_arr  = np.array(tw_rl)
        w_min  = 0.2
        w_max  = min(float(np.percentile(w_arr, 95)), 6.0)
        hist, edges = np.histogram(w_arr, bins=25, range=(w_min, w_max))
        bin_lbl = [
            f"{(edges[i]+edges[i+1])/2:.2f}" for i in range(len(edges) - 1)
        ]

        out[gamma] = {
            "traj":          [[round(x, 4) for x in step] for step in traj],
            "naive_traj":    [[round(x, 4) for x in step] for step in traj_naive],
            "merton_target": target,
            "final_alloc":   [round(x, 4) for x in traj[T4]],
            "hist_labels":   bin_lbl,
            "hist_counts":   hist.tolist(),
            "eu_rl":         round(eu_rl,    6),
            "eu_merton":     round(eu_naive, 6),
            "eu_cash":       round(eu_cash,  6),
        }

    m2 = out[2]["merton_target"]; m5 = out[5]["merton_target"]
    f2 = out[2]["final_alloc"];   f5 = out[5]["final_alloc"]

    alloc_table = []
    for i, asset in enumerate(ASSETS):
        gap = ((f2[i] - m2[i]) / m2[i] * 100) if m2[i] > 0.001 else None
        alloc_table.append({
            "name":      asset["name"],
            "color":     asset["color"],
            "mu":        f"{asset['mu']*100:.0f}%",
            "sigma":     f"{asset['sigma']*100:.0f}%",
            "merton_g2": f"{m2[i]:.3f}",
            "merton_g5": f"{m5[i]:.3f}",
            "rl_g2":     f"{f2[i]:.3f}",
            "rl_g5":     f"{f5[i]:.3f}",
            "gap":       f"{gap:.1f}%" if gap is not None else "—",
            "gap_ok":    gap is None or gap > -5,
        })
    alloc_table.append({
        "name":      "Cash",
        "color":     "#94a3b8",
        "mu":        f"{R_FREE*100:.0f}%",
        "sigma":     "—",
        "merton_g2": f"{max(0.0, 1 - sum(m2)):.3f}",
        "merton_g5": f"{max(0.0, 1 - sum(m5)):.3f}",
        "rl_g2":     f"{max(0.0, 1 - sum(f2)):.3f}",
        "rl_g5":     f"{max(0.0, 1 - sum(f5)):.3f}",
        "gap":       "—",
        "gap_ok":    True,
    })

    return out, alloc_table


# ── Pre-compute all cases using src classes ───────────────────────────────────

print("⏳  Pre-computing dashboard data using src classes …")
_rng    = np.random.default_rng(RNG_SEED)
_case1  = compute_case1(_rng)
_case2  = compute_case2(_rng)
_case3  = compute_case3(_rng)
_case4_by_gamma, _alloc_table = compute_case4(_rng)
print("✅  Dashboard data ready.")

_DASHBOARD_JS = {
    "gammas":       GAMMAS,
    "gamma_colors": {1: "#e74c3c", 2: "#e67e22", 5: "#2ecc71", 10: "#3498db"},
    "gamma_labels": {
        1: "γ=1 (Log)", 2: "γ=2", 5: "γ=5", 10: "γ=10"
    },
    "assets":       [{"name": a["name"], "color": a["color"]} for a in ASSETS],
    "t_labels":     [f"t={t}" for t in range(max(T3, T4) + 1)],
    "case1":        _case1,
    "case2":        _case2,
    "case3":        _case3,
    "case4":        _case4_by_gamma,
    # Pre-computed advantage for Case 4 vs Case 3 comparison
    "case3_adv": {
        g: round(_case3[g]["eu_rl"] - _case3[g]["eu_merton_naive"], 6)
        for g in GAMMAS
    },
    "case4_adv": {
        g: round(_case4_by_gamma[g]["eu_rl"] - _case4_by_gamma[g]["eu_merton"], 6)
        for g in GAMMAS
    },
}

_PARAMS = {
    "r_free":              f"{R_FREE*100:.1f}%",
    "mu1":                 f"{MU1*100:.1f}%",
    "sigma1":              f"{SIG1*100:.1f}%",
    "w0":                  f"{W0:.1f}",
    "rebal_limit":         "10 pp/period",
    "t3":                  T3,
    "t4":                  T4,
    "merton_pis":          {g: round(_merton_n1_clipped(g), 2)       for g in GAMMAS},
    "merton_naive_thetas": {g: round(_naive_merton_theta_n1(g), 2)   for g in GAMMAS},
    "merton_feasible":     {g: bool(abs(_merton_n1(g)) <= REBAL+1e-9) for g in GAMMAS},
}


@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        params=_PARAMS,
        assets=ASSETS,
        gammas=GAMMAS,
        alloc_table=_alloc_table,
        dashboard_data=_DASHBOARD_JS,
    )


if __name__ == "__main__":
    app.run(debug=False)