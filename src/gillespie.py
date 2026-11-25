"""Gillespie (stochastic simulation algorithm) cross-check of the master equation.

The same five-state scheme is simulated one cell at a time; the fraction of
cells absorbed in A must agree with Equation 1.  The implementation is
vectorised over cells: at each iteration every still-unabsorbed cell draws its
next event, so the cost scales with the number of binding/unbinding cycles
rather than with the number of cells.
"""

from __future__ import annotations

import numpy as np

from parameters import Parameters, SEED, default_parameters

#: Integer codes for the cell state during the simulation.
UNBOUND, BOUND_A, BOUND_B, FINAL_A, FINAL_B = range(5)


def simulate(R: float,
             n_cells: int = 100_000,
             p: Parameters | None = None,
             seed: int = SEED,
             t_max: float = np.inf,
             max_iterations: int = 100_000):
    """Run the SSA for ``n_cells`` independent cells at fixed ``R`` (nM).

    Parameters
    ----------
    t_max
        Optional observation time in seconds.  Cells that have not reached A
        or B by then are reported as undifferentiated.  Defaults to running
        until every cell is absorbed.

    Returns
    -------
    dict with keys ``n_A``, ``n_B``, ``n_undifferentiated``, ``P_A``,
    ``P_B`` (fractions among the cells that did differentiate),
    ``mean_time_to_commit`` (seconds, over committed cells only).
    """
    p = default_parameters() if p is None else p
    rng = np.random.default_rng(seed)

    state = np.full(n_cells, UNBOUND, dtype=np.int8)
    time = np.zeros(n_cells, dtype=float)

    s = p.k_on * R  # binding rate to one site
    total_A = p.k_off_A + p.k_cat_A
    total_B = p.k_off_B + p.k_cat_B

    for _ in range(max_iterations):
        active = np.flatnonzero((state <= BOUND_B) & (time < t_max))
        if active.size == 0:
            break

        st = state[active]

        # --- unbound cells: two competing binding reactions, rate s each ----
        idx = active[st == UNBOUND]
        if idx.size:
            if s <= 0.0:
                time[idx] = np.inf  # no recombinase: the cell never leaves 0
            else:
                time[idx] += rng.exponential(1.0 / (2.0 * s), idx.size)
                to_A = rng.random(idx.size) < 0.5
                state[idx] = np.where(to_A, BOUND_A, BOUND_B)

        # --- bound cells: dissociation vs catalysis -------------------------
        for bound, final, k_off, k_cat, total in (
            (BOUND_A, FINAL_A, p.k_off_A, p.k_cat_A, total_A),
            (BOUND_B, FINAL_B, p.k_off_B, p.k_cat_B, total_B),
        ):
            idx = active[st == bound]
            if not idx.size:
                continue
            time[idx] += rng.exponential(1.0 / total, idx.size)
            catalyses = rng.random(idx.size) < (k_cat / total)
            state[idx] = np.where(catalyses, final, UNBOUND)
    else:
        raise RuntimeError(
            "Gillespie did not absorb every cell within max_iterations; "
            "raise max_iterations or set a finite t_max."
        )

    # Cells whose last event carried them past t_max did not commit in time.
    committed = (state >= FINAL_A) & (time <= t_max)
    n_A = int(np.count_nonzero(committed & (state == FINAL_A)))
    n_B = int(np.count_nonzero(committed & (state == FINAL_B)))
    n_committed = n_A + n_B

    return {
        "n_cells": n_cells,
        "n_A": n_A,
        "n_B": n_B,
        "n_undifferentiated": n_cells - n_committed,
        "P_A": n_A / n_committed if n_committed else float("nan"),
        "P_B": n_B / n_committed if n_committed else float("nan"),
        "mean_time_to_commit": (
            float(time[committed].mean()) if n_committed else float("nan")
        ),
    }
