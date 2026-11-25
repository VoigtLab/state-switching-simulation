"""Generate every model result as one spreadsheet.

Writes ``results/model_results.xlsx``.  Each data sheet is a plain rectangular
block with a single header row and numbers below it, so a sheet can be pasted
straight into plotting software.  Sheet contents are listed on the ``contents``
sheet and described in the README.

Run with ``python make_results.py``, or through ``bash run_all.sh``.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from scipy.linalg import expm as scipy_expm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import gillespie  # noqa: E402
import langevin  # noqa: E402
import sensitivity as sens  # noqa: E402
import thresholds as th  # noqa: E402
from model import (  # noqa: E402
    P_A_analytic,
    P_A_from_phi,
    _expm_batch,
    generator_matrix,
    integrate_master_along_R,
    phi,
    solve_master,
    tau,
    tau_exact_mfpt,
    tau_limit_infinite_R,
)
from parameters import (  # noqa: E402
    BETA_0,
    DT,
    GAMMA,
    K_CAT_A,
    K_CAT_A_UNEQUAL,
    K_CAT_B,
    K_CAT_B_UNEQUAL,
    K_D,
    K_OFF_A,
    K_OFF_B,
    K_ON,
    R_INITIAL,
    R_SS,
    SEED,
    SIGMA_NOISELESS,
    SIGMA_NOISY,
    SIGMA_SWEEP,
    T_END,
    default_parameters,
    from_KD,
    unequal_rate_parameters,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
WORKBOOK_NAME = "model_results.xlsx"

# --- Fig. 1C ---------------------------------------------------------------
N_SWEEP_SAMPLES = 4000        # random (K_D, k_cat) draws
K_D_SWEEP_NM = (1.0, 1000.0)  # 1 nM to 1 uM
K_CAT_SWEEP_S = (1e-4, 0.1)   # s^-1
PHI_AXIS = (1e-5, 1e5)        # range of the analytic curve
N_PHI_POINTS = 1000

# --- Fig. 1D ---------------------------------------------------------------
R_AXIS_NM = (0.01, 240.0)
N_R_POINTS = 400

# --- Fig. 1E ---------------------------------------------------------------
N_TRAJECTORIES = 1000         # Langevin trajectories per sigma

# --- sensitivity analysis --------------------------------------------------
R_INVARIANCE_NM = (0.01, 1000.0)   # phi against R
RATE_SCALE_RANGE = (0.01, 100.0)   # common factor on all four rates
K_ON_RANGE = (1e-4, 1e-2)          # nM^-1 s^-1
K_D_RANGE = (1.0, 1000.0)          # nM
N_GRID = 41
R_LOW_RANGE_NM = (0.01, 50.0)
N_R_LOW = 200

# --- Gillespie cross-check -------------------------------------------------
N_GILLESPIE_CELLS = 100_000


def _logspace(bounds, n):
    return np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), n)


# ==========================================================================
# Workbook helpers
# ==========================================================================

_BOLD = Font(bold=True)


def _add_table(wb, title, header, columns, number_format="0.000000000000"):
    """Write one sheet: a bold header row, then the columns as numbers."""
    ws = wb.create_sheet(title)
    ws.append(list(header))
    for cell in ws[1]:
        cell.font = _BOLD
    rows = np.column_stack([np.asarray(c, dtype=float) for c in columns])
    for row in rows:
        ws.append([float(v) for v in row])
    for col in range(1, len(header) + 1):
        letter = get_column_letter(col)
        ws.column_dimensions[letter].width = max(12, len(header[col - 1]) + 2)
        for cell in ws[letter][1:]:
            cell.number_format = number_format
    ws.freeze_panes = "A2"
    return ws


def _add_text_sheet(wb, title, rows, widths=(34, 86)):
    """Write one sheet of label/value text, for metadata rather than data."""
    ws = wb.create_sheet(title)
    for row in rows:
        ws.append(list(row))
    for cell in ws[1]:
        cell.font = _BOLD
    for i, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width
    return ws


# ==========================================================================
# Sheets
# ==========================================================================

def sheet_contents(wb):
    _add_text_sheet(
        wb,
        "contents",
        [
            ("sheet", "what it holds"),
            ("parameters", "every model parameter, with units and source"),
            ("Fig1C_curve", "analytic P_A and P_B against the outcome ratio phi (Equation 1)"),
            ("Fig1C_sweep", "phi and P_A for 4000 random (K_D, k_cat) parameter draws"),
            ("Fig1D_tau", "commitment timescale tau against recombinase concentration R (Equation 2)"),
            ("Fig1E_traces", "one recombinase trajectory per noise condition over 240 min (Equation 3)"),
            ("Fig1E_noise_sweep", "P_A, P_B and the undifferentiated fraction at 240 min against noise amplitude sigma"),
            ("Fig1E_fluctuations", "spread of R across cells at 240 min for each sigma: median, percentiles, IQR, SD"),
            ("SensA_invariance", "phi against R, and phi under a common rescaling of all four rate constants"),
            ("SensB_phi_grid", "phi over a 41 x 41 grid of k_on and K_D"),
            ("SensC_undiff_curve", "undifferentiated fraction against R at 6 h and 24 h, two ways"),
            ("SensC_undiff_table", "the same at five discrete R values"),
            ("SensD_ratio", "p_A/(p_A+p_B) among committed cells against R"),
            ("Thresh_summary", "for each division time, the R, k_cat and K_D at which the switch stops outpacing division (k_on from the exact MFPT)"),
            ("Thresh_vs_t_div", "the same thresholds as continuous curves against division time"),
            ("Thresh_dimensionless", "the boundary in dimensionless form: minimum k_cat*t_div against K_D/R"),
            ("validation", "each check, its expected value, the computed value, and pass/fail"),
            ("", ""),
            ("note", "Data sheets are header row plus numbers only, so a block pastes directly into plotting software."),
            ("", "Every value is computed from the model equations by make_results.py."),
            ("", f"All stochastic results use numpy.random.default_rng seeded from {SEED}, so the workbook is reproducible."),
        ],
    )


def sheet_parameters(wb):
    _add_text_sheet(
        wb,
        "parameters",
        [
            ("parameter", "value", "units", "source"),
            ("k_on", K_ON, "nM^-1 s^-1", "estimated from crowding / non-facilitated target search"),
            ("K_D", K_D, "nM", "literature value for Bxb1 binding affinity"),
            ("k_off_A", K_OFF_A, "s^-1", "K_D * k_on"),
            ("k_off_B", K_OFF_B, "s^-1", "K_D * k_on"),
            ("k_cat_A", K_CAT_A, "s^-1", "equal rates; even A/B partitioning"),
            ("k_cat_B", K_CAT_B, "s^-1", "equal rates; even A/B partitioning"),
            ("beta_0", BETA_0, "nM s^-1", "recombinase production; burst size ~7"),
            ("gamma", GAMMA, "s^-1", "dilution by growth, 77 min doubling"),
            ("R_ss", R_SS, "nM", "beta_0 / gamma"),
            ("R(0)", R_INITIAL, "nM", "no recombinase at induction"),
            ("sigma (noiseless)", SIGMA_NOISELESS, "s^-1/2", "reference condition"),
            ("sigma (noisy)", SIGMA_NOISY, "s^-1/2", "Fig. 1E upper trace"),
            ("dt", DT, "s", "Euler-Maruyama timestep"),
            ("t_end", T_END, "s", "240 min observation window"),
            ("seed", SEED, "-", "numpy.random.default_rng"),
            ("", "", "", ""),
            ("k_cat_A (unequal-rate example)", K_CAT_A_UNEQUAL, "s^-1", "biased example, phi != 1; not the switch characterised here"),
            ("k_cat_B (unequal-rate example)", K_CAT_B_UNEQUAL, "s^-1", "biased example, phi != 1; not the switch characterised here"),
            ("", "", "", ""),
            ("note", "Recombinase abundance is expressed in nM; for E. coli one molecule per cell corresponds to approximately 1 nM.", "", ""),
        ],
        widths=(24, 16, 14, 70),
    )


def sheet_fig1C(wb):
    rng = np.random.default_rng(SEED)

    def loguniform(low, high, n):
        return np.exp(rng.uniform(np.log(low), np.log(high), n))

    K_D_A = loguniform(*K_D_SWEEP_NM, N_SWEEP_SAMPLES)
    K_D_B = loguniform(*K_D_SWEEP_NM, N_SWEEP_SAMPLES)
    k_cat_A = loguniform(*K_CAT_SWEEP_S, N_SWEEP_SAMPLES)
    k_cat_B = loguniform(*K_CAT_SWEEP_S, N_SWEEP_SAMPLES)

    phi_sweep = np.array(
        [
            phi(from_KD(K_ON, a, b, ca, cb))
            for a, b, ca, cb in zip(K_D_A, K_D_B, k_cat_A, k_cat_B)
        ]
    )
    P_A_sweep = P_A_from_phi(phi_sweep)

    phi_curve = _logspace(PHI_AXIS, N_PHI_POINTS)
    P_A_curve = P_A_from_phi(phi_curve)

    _add_table(wb, "Fig1C_curve", ("phi", "P_A", "P_B"),
               (phi_curve, P_A_curve, 1.0 - P_A_curve))
    _add_table(
        wb, "Fig1C_sweep",
        ("K_D_A_nM", "K_D_B_nM", "k_cat_A_per_s", "k_cat_B_per_s", "phi", "P_A"),
        (K_D_A, K_D_B, k_cat_A, k_cat_B, phi_sweep, P_A_sweep),
    )
    return {
        "phi_sweep_min": float(phi_sweep.min()),
        "phi_sweep_max": float(phi_sweep.max()),
    }


def sheet_fig1D(wb):
    R = _logspace(R_AXIS_NM, N_R_POINTS)
    _add_table(wb, "Fig1D_tau",
               ("R_nM", "tau_s", "tau_exact_mfpt_s"),
               (R, tau(R), tau_exact_mfpt(R)))
    return {
        "tau_at_R_1_s": float(tau(1.0)),
        "tau_at_R_100_s": float(tau(100.0)),
        "tau_infinite_R_s": tau_limit_infinite_R(),
    }


def sheet_fig1E_traces(wb):
    t, R_det = langevin.simulate(SIGMA_NOISELESS, n_traj=1, seed=SEED)
    _, R_noisy = langevin.simulate(SIGMA_NOISY, n_traj=1, seed=SEED)
    _add_table(
        wb, "Fig1E_traces",
        ("time_s", "time_min", "R_sigma_0_nM", "R_sigma_0p064_nM"),
        (t, t / 60.0, R_det[0], R_noisy[0]),
    )
    return {
        "R_sigma0_at_t_end_nM": float(R_det[0, -1]),
        "R_sigma0_closed_form_nM": float(langevin.deterministic_solution(t[-1])),
    }


def sheet_fig1E_sweep(wb, n_traj=N_TRAJECTORIES, verbose=True):
    """P_A, P_B and the undifferentiated fraction at t_end against sigma.

    For each sigma: ``n_traj`` independent Langevin trajectories, then the
    five-state master equation integrated along each cell's own R(t).  The
    reported value is the mean of p_A(t_end) over cells.
    """
    P_A = np.empty(SIGMA_SWEEP.size)
    P_B = np.empty(SIGMA_SWEEP.size)
    undiff = np.empty(SIGMA_SWEEP.size)
    for i, sigma in enumerate(SIGMA_SWEEP):
        started = time.time()
        # A distinct but deterministic seed per sigma keeps the conditions
        # independent while leaving the workbook reproducible from SEED.
        _, R_traj = langevin.simulate(float(sigma), n_traj=n_traj, seed=SEED + i)
        y_end = integrate_master_along_R(R_traj, DT)
        P_A[i] = y_end[:, 3].mean()
        P_B[i] = y_end[:, 4].mean()
        undiff[i] = y_end[:, :3].sum(axis=1).mean()
        if verbose:
            print(f"  sigma = {sigma:.3f}  P_A = {P_A[i]:.12f}  "
                  f"P_B = {P_B[i]:.12f}  |P_A - P_B| = {abs(P_A[i] - P_B[i]):.2e}  "
                  f"undifferentiated = {undiff[i]:.9f}  "
                  f"({time.time() - started:.1f} s)", flush=True)

    # p_A + p_B falls just short of 1 at large sigma: those cells have not
    # committed by t_end.  Reported as computed, not normalised away.
    _add_table(
        wb, "Fig1E_noise_sweep",
        ("sigma_per_sqrt_s", "P_A", "P_B", "undifferentiated_fraction",
         "ratio_P_A_over_P_A_plus_P_B"),
        (SIGMA_SWEEP, P_A, P_B, undiff, P_A / (P_A + P_B)),
    )
    return {
        "P_A_at_sigma_0": float(P_A[0]),
        "P_A_at_sigma_max": float(P_A[-1]),
        "max_abs_P_A_minus_P_B": float(np.abs(P_A - P_B).max()),
        "max_undifferentiated": float(undiff.max()),
        "max_ratio_deviation": float(
            np.abs(P_A / (P_A + P_B) - 0.5).max()
        ),
        # Expressed as a total decrease so that 0 means "rises monotonically".
        "undiff_total_decrease_over_sigma": float(
            -np.minimum(np.diff(undiff), 0.0).sum()
        ),
    }


def sheet_fluctuations(wb, n_traj=N_TRAJECTORIES):
    """Spread of R across cells at t_end, for every sigma plus sigma = 0.064.

    This is what "the magnitude of the fluctuations" means for this process.
    Under multiplicative noise the stationary density of Equation 3 is
    inverse-gamma with shape ``1 + 2*gamma/sigma**2``; its variance is finite
    only while ``sigma < sqrt(2*gamma)`` = 0.01732 s^-1/2.  Above that the
    standard deviation does not converge as cells are added or the run is
    lengthened, so percentiles, not the SD, are the statistic to compare
    between two realisations.  Both are tabulated, with the SD alongside for
    reference.
    """
    sigma_values = np.union1d(SIGMA_SWEEP, [SIGMA_NOISY])
    columns = {k: np.empty(sigma_values.size) for k in
               ("mean", "sd", "cv", "p5", "p25", "median", "p75", "p95",
                "min", "max", "iqr_over_median")}

    for i, sigma in enumerate(sigma_values):
        # Same seeding rule as the noise sweep, so a sigma on the sweep grid
        # gets the same trajectories there and here.
        index = int(np.argmin(np.abs(SIGMA_SWEEP - sigma)))
        seed = SEED + index if np.isclose(SIGMA_SWEEP[index], sigma) else SEED
        _, R_traj = langevin.simulate(float(sigma), n_traj=n_traj, seed=seed)
        final = R_traj[:, -1]
        columns["mean"][i] = final.mean()
        columns["sd"][i] = final.std(ddof=1)
        columns["cv"][i] = columns["sd"][i] / columns["mean"][i]
        for key, q in (("p5", 5), ("p25", 25), ("median", 50),
                       ("p75", 75), ("p95", 95)):
            columns[key][i] = np.percentile(final, q)
        columns["min"][i] = final.min()
        columns["max"][i] = final.max()
        columns["iqr_over_median"][i] = (
            (columns["p75"][i] - columns["p25"][i]) / columns["median"][i]
        )

    _add_table(
        wb, "Fig1E_fluctuations",
        ("sigma_per_sqrt_s", "R_mean_nM", "R_sd_nM", "R_cv", "R_p5_nM",
         "R_p25_nM", "R_median_nM", "R_p75_nM", "R_p95_nM", "R_min_nM",
         "R_max_nM", "R_iqr_over_median"),
        (sigma_values, columns["mean"], columns["sd"], columns["cv"],
         columns["p5"], columns["p25"], columns["median"], columns["p75"],
         columns["p95"], columns["min"], columns["max"],
         columns["iqr_over_median"]),
    )

    at_noisy = int(np.argmin(np.abs(sigma_values - SIGMA_NOISY)))
    return {
        "sigma_finite_variance_limit": float(np.sqrt(2.0 * GAMMA)),
        "R_median_at_sigma_noisy": float(columns["median"][at_noisy]),
        "R_iqr_over_median_at_sigma_noisy": float(
            columns["iqr_over_median"][at_noisy]
        ),
        "R_cv_at_sigma_noisy": float(columns["cv"][at_noisy]),
        "R_spread_at_sigma_0": float(columns["sd"][0]),
    }


def sheet_sensitivity(wb):
    R_inv = _logspace(R_INVARIANCE_NM, 200)
    scales = _logspace(RATE_SCALE_RANGE, 200)
    phi_R = sens.phi_versus_R(R_inv)
    phi_scaled = sens.phi_under_rate_rescaling(scales)
    _add_table(wb, "SensA_invariance",
               ("R_nM_or_rate_scale", "phi_vs_R", "phi_vs_rate_scale"),
               (R_inv, phi_R, phi_scaled))

    k_on_values = _logspace(K_ON_RANGE, N_GRID)
    K_D_values = _logspace(K_D_RANGE, N_GRID)
    grid = sens.phi_grid(k_on_values, K_D_values)
    # One row per K_D, one column per k_on, with the axis values in the margins.
    ws = wb.create_sheet("SensB_phi_grid")
    ws.append(["K_D_nM \\ k_on_nM-1s-1"] + [float(v) for v in k_on_values])
    for cell in ws[1]:
        cell.font = _BOLD
    for j, K_D_value in enumerate(K_D_values):
        ws.append([float(K_D_value)] + [float(v) for v in grid[j]])
    ws.column_dimensions["A"].width = 24
    ws.freeze_panes = "B2"

    R_low = _logspace(R_LOW_RANGE_NM, N_R_LOW)
    six_h = sens.OBSERVATION_TIMES_S["6 h"]
    day = sens.OBSERVATION_TIMES_S["24 h"]
    _add_table(
        wb, "SensC_undiff_curve",
        ("R_nM", "tau_s", "undiff_6h_single_exponential",
         "undiff_24h_single_exponential", "undiff_6h_master_equation",
         "undiff_24h_master_equation"),
        (R_low, tau(R_low),
         sens.undifferentiated_curve(R_low, six_h),
         sens.undifferentiated_curve(R_low, day),
         sens.undifferentiated_exact(R_low, six_h),
         sens.undifferentiated_exact(R_low, day)),
    )

    table = sens.undifferentiated_table()
    _add_table(
        wb, "SensC_undiff_table",
        ("R_nM", "tau_s", "tau_h", "undiff_6h", "undiff_24h"),
        ([r["R_nM"] for r in table], [r["tau_s"] for r in table],
         [r["tau_h"] for r in table], [r["undiff_6h"] for r in table],
         [r["undiff_24h"] for r in table]),
    )

    _add_table(
        wb, "SensD_ratio",
        ("R_nM", "ratio_6h", "ratio_24h"),
        (R_low,
         sens.ratio_among_differentiated(R_low, six_h),
         sens.ratio_among_differentiated(R_low, day)),
    )

    report = sens.symbolic_invariance_report()
    return {
        "phi_spread_over_R": float(np.ptp(phi_R)),
        "phi_spread_over_rate_scale": float(np.ptp(phi_scaled)),
        "phi_grid_spread": float(np.ptp(grid)),
        "dphi_dR_symbolic": str(report["dphi_dR"]),
        "rescaling_difference_symbolic": str(report["difference"]),
    }


def sheet_thresholds(wb):
    """Where the switch stops keeping up with cell division."""
    rows = th.threshold_table()

    ws = wb.create_sheet("Thresh_summary")
    ws.append(["division time", "t_div_s", "t_div_min", "tau_at_R_ss_s",
               "tau_over_t_div", "R_threshold_nM", "R_margin_fold",
               "k_cat_threshold_per_s", "k_cat_margin_fold",
               "K_D_threshold_nM", "K_D_margin_fold",
               "k_on_threshold_exact_per_nM_s", "k_on_margin_fold",
               "turnovers_required", "turnovers_available"])
    for cell in ws[1]:
        cell.font = _BOLD
    for row in rows:
        point = th.operating_point(row["t_div_s"])
        ws.append([
            row["label"], row["t_div_s"], row["t_div_min"],
            row["tau_at_R_ss_s"], row["tau_at_R_ss_s"] / row["t_div_s"],
            row["R_threshold_nM"], row["R_margin"],
            row["k_cat_threshold_per_s"], row["k_cat_margin"],
            row["K_D_threshold_nM"], row["K_D_margin"],
            row["k_on_threshold_exact"], row["k_on_margin"],
            point["turnovers_required"], point["turnovers_available"],
        ])
    ws.column_dimensions["A"].width = 36
    for col in range(2, 16):
        letter = get_column_letter(col)
        ws.column_dimensions[letter].width = 22
        for cell in ws[letter][1:]:
            cell.number_format = "0.000000000000"
    ws.freeze_panes = "B2"

    curves = th.threshold_curves()
    _add_table(
        wb, "Thresh_vs_t_div",
        ("t_div_s", "t_div_min", "R_threshold_nM", "k_cat_threshold_per_s",
         "K_D_threshold_nM", "k_on_threshold_exact_per_nM_s"),
        (curves["t_div_s"], curves["t_div_min"], curves["R_threshold_nM"],
         curves["k_cat_threshold_per_s"], curves["K_D_threshold_nM"],
         curves["k_on_threshold_exact"]),
    )

    ratio, required = th.dimensionless_boundary()
    _add_table(
        wb, "Thresh_dimensionless",
        ("K_D_over_R", "min_k_cat_times_t_div"),
        (ratio, required),
    )

    report = th.symbolic_report()
    point = th.operating_point(th.doubling_time_from_gamma())
    return {
        "tau_reduced_symbolic": report["tau_reduced"],
        "dtau_dk_on_symbolic": report["dtau_dk_on"],
        "k_on_threshold_from_Eq2": report["k_on_threshold_from_Eq2"],
        "R_threshold_at_model_doubling_nM": float(
            th.R_threshold(th.doubling_time_from_gamma())
        ),
        "tau_reduced_max_relative_error": float(
            np.abs(th.tau_reduced(np.logspace(-2, 3, 200))
                   / tau(np.logspace(-2, 3, 200)) - 1.0).max()
        ),
        "operating_point_margin": point["margin"],
        "threshold_consistency_error": max(
            abs(float(tau(r["R_threshold_nM"])) - r["t_div_s"])
            for r in rows
        ),
        "k_on_invariance_error": float(np.abs(
            tau(50.0, from_KD(100 * K_ON, K_D, K_D, K_CAT_A, K_CAT_B))
            - tau(50.0)
        )),
    }


# ==========================================================================
# Validation
# ==========================================================================

def sheet_validation(wb, measured):
    """Recompute every validation target and record expected vs computed.

    This sheet is the repository's verification: each target is an analytic
    value of the model equations, a Monte Carlo estimate of one, or a
    property the implementation must have (conservation, monotonicity,
    reproducibility from the seed).  Properties are expressed as violations,
    so an expected value of 0 means "this does not happen".
    """
    p_default = default_parameters()
    p_unequal = unequal_rate_parameters()

    t_eval = np.linspace(0.0, T_END, 1001)
    y = solve_master(R_SS, t_eval)
    conservation_error = float(np.abs(y.sum(axis=1) - 1.0).max())

    gill = gillespie.simulate(R_SS, N_GILLESPIE_CELLS, seed=SEED)
    gill_unequal = gillespie.simulate(R_SS, N_GILLESPIE_CELLS, seed=SEED,
                                    p=p_unequal)
    # Equation 1 says the outcome is independent of R; the SSA should agree.
    gill_spread_over_R = max(
        abs(gillespie.simulate(R, 20_000, seed=SEED)["P_A"] - 0.5)
        for R in (0.5, 5.0, 500.0)
    )

    table = sens.undifferentiated_table()
    by_R = {row["R_nM"]: row for row in table}

    # Batched propagator against two independent references.
    stack = np.stack([generator_matrix(R) * DT
                      for R in (0.0, 0.01, 1.0, 50.0, 1e3, 1e5)])
    expm_error = float(
        np.abs(_expm_batch(stack)
               - np.stack([scipy_expm(M) for M in stack])).max()
    )
    n_steps = 400
    y_const = integrate_master_along_R(np.full((1, n_steps + 1), R_SS), DT)[0]
    propagator_error = float(
        np.abs(y_const
               - solve_master(R_SS, np.array([0.0, n_steps * DT]))[-1]).max()
    )

    # Shape of the noiseless trajectory, and behaviour along noisy ones.
    t_trace, R_det = langevin.simulate(SIGMA_NOISELESS, n_traj=1, seed=SEED)
    # Both are expressed as violations, so that 0 means the property holds.
    trace_total_decrease = float(
        -np.minimum(np.diff(R_det[0]), 0.0).sum()
    )
    trace_overshoot = max(0.0, float(R_det[0, -1]) - R_SS)
    grid_step_error = float(np.abs(np.diff(t_trace) - DT).max())

    _, R_noisy = langevin.simulate(SIGMA_NOISY, n_traj=100, seed=SEED)
    y_noisy = integrate_master_along_R(R_noisy, DT)
    noisy_conservation = float(np.abs(y_noisy.sum(axis=1) - 1.0).max())
    negative_probability = max(0.0, -float(y_noisy.min()))
    negative_R = max(0.0, -float(R_noisy.min()))

    # The sigma grid, and reproducibility from the seed.
    sigma_spacing_error = float(np.abs(np.diff(SIGMA_SWEEP) - 0.005).max())
    sigma_noisy_on_grid = int(np.isclose(SIGMA_SWEEP, SIGMA_NOISY).sum())
    _, repeat_a = langevin.simulate(SIGMA_NOISY, n_traj=50, seed=SEED)
    _, repeat_b = langevin.simulate(SIGMA_NOISY, n_traj=50, seed=SEED)
    seed_repeat_error = float(np.abs(repeat_a - repeat_b).max())

    # The single-exponential shortcut against the full master equation.
    R_cmp = np.logspace(-2, 0, 20)
    six_h = sens.OBSERVATION_TIMES_S["6 h"]
    single_exponential_gap = float(
        np.abs(sens.undifferentiated_curve(R_cmp, six_h)
               - sens.undifferentiated_exact(R_cmp, six_h)).max()
    )

    checks = [
        ("Eq. 1  phi, current parameters", 1.0, phi(p_default), 1e-12),
        ("Eq. 1  P_A, current parameters", 0.5, P_A_analytic(p_default), 1e-12),
        ("Eq. 1  phi, unequal rates", 4.5752688172043019, phi(p_unequal), 1e-9),
        ("Eq. 1  P_A, unequal rates", 0.8206364513018323, P_A_analytic(p_unequal), 1e-9),
        ("Eq. 1  spread of phi over R in [1e-2, 1e3]", 0.0, measured["phi_spread_over_R"], 1e-15),
        ("Eq. 1  spread of phi under common rate rescaling", 0.0, measured["phi_spread_over_rate_scale"], 1e-12),
        ("Eq. 1  spread of phi over the k_on x K_D grid", 0.0, measured["phi_grid_spread"], 1e-12),
        ("Eq. 2  tau at R = 1 nM (s)", 9000.0, tau(1.0), 1e-9),
        ("Eq. 2  tau at R = 100 nM (s)", 337.5, tau(100.0), 1e-9),
        ("Eq. 2  tau as R -> infinity (s)", 250.0, tau_limit_infinite_R(), 1e-9),
        ("Eq. 2  tau at R = 1 nM, unequal rates (s)", 2666.6666666666665, tau(1.0, p_unequal), 1e-6),
        ("Master equation  probability conservation", 0.0, conservation_error, 1e-9),
        ("Master equation  p_A at R = 50 nM, t_end", 0.5, float(y[-1, 3]), 1e-9),
        ("Master equation  uncommitted at R = 50 nM, t_end", 0.0, float(y[-1, :3].sum()), 1e-9),
        ("Gillespie  P_A, 100000 cells", 0.5, gill["P_A"], 0.01),
        ("Gillespie  P_A, unequal rates", 0.8206364513018323, gill_unequal["P_A"], 0.01),
        ("Gillespie  mean commitment time vs exact MFPT (s)", float(tau_exact_mfpt(R_SS)), gill["mean_time_to_commit"], 5.0),
        ("Gillespie  max |P_A - 0.5| at R = 0.5, 5, 500 nM", 0.0, gill_spread_over_R, 0.01),
        ("Propagator  batched expm vs scipy.linalg.expm", 0.0, expm_error, 1e-10),
        ("Propagator  constant R vs solve_ivp", 0.0, propagator_error, 1e-9),
        ("Propagator  conservation along 100 noisy trajectories", 0.0, noisy_conservation, 1e-9),
        ("Propagator  negative probability along noisy trajectories", 0.0, negative_probability, 1e-12),
        ("Eq. 3  R at t_end, sigma = 0 (nM)", 44.23374394809687, measured["R_sigma0_at_t_end_nM"], 0.05),
        ("Eq. 3  total decrease of the sigma = 0 trace (nM)", 0.0, trace_total_decrease, 0.0),
        ("Eq. 3  overshoot of R_ss by the sigma = 0 trace (nM)", 0.0, trace_overshoot, 0.0),
        ("Eq. 3  timestep deviation from dt (s)", 0.0, grid_step_error, 1e-9),
        ("Eq. 3  negative R anywhere in 100 trajectories (nM)", 0.0, negative_R, 0.0),
        ("Eq. 3  same seed reproduces the trajectory exactly", 0.0, seed_repeat_error, 0.0),
        ("sigma grid  number of points", 16.0, float(SIGMA_SWEEP.size), 0.0),
        ("sigma grid  spacing deviation from 0.005", 0.0, sigma_spacing_error, 1e-12),
        ("sigma grid  grid points equal to sigma = 0.064", 0.0, float(sigma_noisy_on_grid), 0.0),
        ("Fig. 1E  P_A at sigma = 0", 0.5, measured["P_A_at_sigma_0"], 1e-9),
        ("Fig. 1E  max |P_A - P_B| over all sigma", 0.0, measured["max_abs_P_A_minus_P_B"], 1e-12),
        ("Fig. 1E  max |p_A/(p_A+p_B) - 0.5| over all sigma", 0.0, measured["max_ratio_deviation"], 1e-12),
        ("Fig. 1E  total decrease of undifferentiated fraction with sigma", 0.0, measured["undiff_total_decrease_over_sigma"], 0.0),
        ("Units  k_on/k_off_A - 1/K_D (nM^-1)", 0.0, abs(K_ON / K_OFF_A - 1.0 / K_D), 1e-15),
        ("Units  t_end in minutes", 240.0, T_END / 60.0, 1e-12),
        ("Sensitivity  single exponential vs master equation", 0.0, single_exponential_gap, 0.03),
        ("Thresholds  reduced tau vs Equation 2 (relative)", 0.0, measured["tau_reduced_max_relative_error"], 1e-12),
        ("Thresholds  tau is unchanged by a 100x change in k_on at fixed K_D", 0.0, measured["k_on_invariance_error"], 1e-12),
        ("Thresholds  tau(R*) - t_div across all division times (s)", 0.0, measured["threshold_consistency_error"], 1e-6),
        ("Thresholds  R* at the model's own doubling time (nM)", 2.0018, measured["R_threshold_at_model_doubling_nM"], 1e-3),
        ("Thresholds  operating-point margin, turnovers available/required", 10.87, measured["operating_point_margin"], 0.01),
        ("Fluctuations  sigma below which the stationary variance is finite", 0.017320508075688773, measured["sigma_finite_variance_limit"], 1e-15),
        ("Fluctuations  spread of R across cells at sigma = 0 (nM)", 0.0, measured["R_spread_at_sigma_0"], 1e-12),
        ("Sensitivity  undifferentiated at R = 0.05 nM, 6 h", 0.884, by_R[0.05]["undiff_6h"], 5e-4),
        ("Sensitivity  undifferentiated at R = 0.05 nM, 24 h", 0.611, by_R[0.05]["undiff_24h"], 5e-4),
        ("Sensitivity  undifferentiated at R = 0.1 nM, 6 h", 0.782, by_R[0.1]["undiff_6h"], 5e-4),
        ("Sensitivity  undifferentiated at R = 0.1 nM, 24 h", 0.374, by_R[0.1]["undiff_24h"], 5e-4),
        ("Sensitivity  undifferentiated at R = 0.3 nM, 6 h", 0.480, by_R[0.3]["undiff_6h"], 5e-4),
        ("Sensitivity  undifferentiated at R = 0.3 nM, 24 h", 0.053, by_R[0.3]["undiff_24h"], 5e-4),
        ("Sensitivity  undifferentiated at R = 1 nM, 6 h", 0.091, by_R[1.0]["undiff_6h"], 5e-4),
        ("Sensitivity  undifferentiated at R = 1 nM, 24 h", 0.000, by_R[1.0]["undiff_24h"], 5e-4),
        ("Sensitivity  undifferentiated at R = 50 nM, 6 h", 0.000, by_R[50.0]["undiff_6h"], 5e-4),
        ("Sensitivity  undifferentiated at R = 50 nM, 24 h", 0.000, by_R[50.0]["undiff_24h"], 5e-4),
    ]

    ws = wb.create_sheet("validation")
    ws.append(["target", "expected", "computed", "abs. difference",
               "tolerance", "result"])
    for cell in ws[1]:
        cell.font = _BOLD
    n_failed = 0
    for name, expected, computed, tolerance in checks:
        difference = abs(float(computed) - float(expected))
        passed = difference <= tolerance
        n_failed += not passed
        ws.append([name, float(expected), float(computed), difference,
                   tolerance, "PASS" if passed else "FAIL"])
    for letter, width in (("A", 52), ("B", 20), ("C", 20), ("D", 16),
                          ("E", 12), ("F", 9)):
        ws.column_dimensions[letter].width = width
    for letter in ("B", "C", "D", "E"):
        for cell in ws[letter][1:]:
            cell.number_format = "0.000000000000"
    ws.freeze_panes = "A2"

    ws.append([])
    ws.append(["symbolic: d(phi)/dR", measured["dphi_dR_symbolic"]])
    ws.append(["symbolic: reduced tau", measured["tau_reduced_symbolic"]])
    ws.append(["symbolic: d(tau)/d(k_on)", measured["dtau_dk_on_symbolic"]])
    ws.append(["symbolic: k_on threshold from Equation 2",
               measured["k_on_threshold_from_Eq2"]])
    ws.append(["symbolic: phi(c*k) - phi(k)",
               measured["rescaling_difference_symbolic"]])
    return len(checks), n_failed


# ==========================================================================
# Entry point
# ==========================================================================

def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    started = time.time()

    wb = Workbook()
    wb.remove(wb.active)          # drop the default empty sheet

    sheet_contents(wb)
    sheet_parameters(wb)

    measured = {}
    print("Equation 1, outcome ratio ...", flush=True)
    measured.update(sheet_fig1C(wb))
    print("Equation 2, commitment timescale ...", flush=True)
    measured.update(sheet_fig1D(wb))
    print("Equation 3, recombinase trajectories ...", flush=True)
    measured.update(sheet_fig1E_traces(wb))
    print("Noise sweep, master equation along each trajectory ...", flush=True)
    measured.update(sheet_fig1E_sweep(wb))
    print("Fluctuation magnitude ...", flush=True)
    measured.update(sheet_fluctuations(wb))
    print("Sensitivity analysis ...", flush=True)
    measured.update(sheet_sensitivity(wb))
    print("Division-time thresholds ...", flush=True)
    measured.update(sheet_thresholds(wb))
    print("Validation ...", flush=True)
    n_checks, n_failed = sheet_validation(wb, measured)

    # Put the metadata sheets first, data after, validation last.
    order = ["contents", "parameters", "Fig1C_curve", "Fig1C_sweep",
             "Fig1D_tau", "Fig1E_traces", "Fig1E_noise_sweep",
             "Fig1E_fluctuations", "SensA_invariance", "SensB_phi_grid",
             "SensC_undiff_curve", "SensC_undiff_table", "SensD_ratio",
             "Thresh_summary", "Thresh_vs_t_div", "Thresh_dimensionless",
             "validation"]
    wb._sheets = [wb[name] for name in order]

    path = os.path.join(OUTPUT_DIR, WORKBOOK_NAME)
    wb.save(path)

    print()
    print(f"wrote {os.path.abspath(path)}  ({len(order)} sheets)")
    print(f"validation: {n_checks - n_failed}/{n_checks} passed")
    print(f"phi spread over R: {measured['phi_spread_over_R']:.3e}")
    print(f"R(t_end), sigma = 0: {measured['R_sigma0_at_t_end_nM']:.9f} nM "
          f"(closed form {measured['R_sigma0_closed_form_nM']:.9f})")
    print(f"P_A at sigma = 0: {measured['P_A_at_sigma_0']:.12f}")
    print(f"P_A at sigma = {SIGMA_SWEEP[-1]:.3f}: "
          f"{measured['P_A_at_sigma_max']:.12f}")
    print(f"max |P_A - P_B| over sigma: "
          f"{measured['max_abs_P_A_minus_P_B']:.3e}")
    print(f"max undifferentiated fraction: "
          f"{measured['max_undifferentiated']:.9f}")
    print(f"total {time.time() - started:.1f} s")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
