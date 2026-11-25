# Model code: stochastic differentiation of bacteria

Source code for the model and the data plotted in Figures 1C–1E and the
parameter sensitivity analysis of *Programmable stochastic differentiation of
bacteria*.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` checks installed package versions against `requirements.txt`, then
writes `results/model_results.xlsx`. Runtime is about three minutes. It exits
non-zero if the versions do not match or if any check on the `validation` sheet
fails. `python make_results.py` runs the same work without the version check.

All random draws come from `numpy.random.default_rng` seeded from
`SEED = 2025` in `src/parameters.py`. Two runs on the same versions give
identical numbers.

## Output

`results/model_results.xlsx`, seventeen sheets. A generated copy is included
here. Each data sheet is one header row followed by numbers.

| sheet | contents |
|---|---|
| `contents` | index of the workbook |
| `parameters` | every parameter, with units and source |
| `Fig1C_curve` | analytic `P_A`, `P_B` against φ, 1000 points from 1e-5 to 1e5 |
| `Fig1C_sweep` | φ and `P_A` for 4000 random (`K_D`, `k_cat`) draws |
| `Fig1D_tau` | τ against `R`, 400 points from 0.01 to 240 nM, with the exact MFPT |
| `Fig1E_traces` | `R(t)` for σ = 0 and σ = 0.064, 7201 points over 240 min |
| `Fig1E_noise_sweep` | `P_A`, `P_B`, undifferentiated fraction and committed-cell ratio at each σ |
| `Fig1E_fluctuations` | spread of `R` across cells at 240 min for each σ |
| `SensA_invariance` | φ against `R`, and against a common rescaling of all four rates |
| `SensB_phi_grid` | φ over a 41 × 41 grid of `k_on` × `K_D` |
| `SensC_undiff_curve` | undifferentiated fraction against `R` at 6 h and 24 h |
| `SensC_undiff_table` | the same at five discrete `R` values |
| `SensD_ratio` | `p_A/(p_A + p_B)` among committed cells against `R` |
| `Thresh_summary` | `R`, `k_cat` and `K_D` thresholds for each division time |
| `Thresh_vs_t_div` | the same thresholds against division time |
| `Thresh_dimensionless` | minimum `k_cat·t_div` against `K_D/R` |
| `validation` | every check: expected, computed, difference, tolerance, result |

No figures are produced.

## The model

A single-copy genetic switch in one bacterium. Recombinase `R` binds one of two
mutually exclusive att sites and then either inverts (path A) or excises
(path B) the DNA. Five states: `0` unbound, `IA`/`IB` bound intermediates,
`A`/`B` absorbing final states. The state vector is ordered
`[p_0, p_IA, p_IB, p_A, p_B]`.

### Master equations

```
dp_0 /dt = -2*k_on*R*p_0 + k_off_A*p_IA + k_off_B*p_IB
dp_IA/dt =    k_on*R*p_0 - (k_off_A + k_cat_A)*p_IA
dp_IB/dt =    k_on*R*p_0 - (k_off_B + k_cat_B)*p_IB
dp_A /dt = k_cat_A*p_IA
dp_B /dt = k_cat_B*p_IB
```

The intermediates `p_IA`/`p_IB` appear on the right-hand sides; `p_A` and `p_B`
do not. Initial condition `p_0(0) = 1`, everything else 0. Total probability
holds to ~1e-14, and is checked to 1e-9 at constant `R` and along 100 noisy
trajectories.

### Equation 1, outcome ratio

```
phi = P_A/P_B = (k_cat_A*(k_off_B + k_cat_B)) / (k_cat_B*(k_off_A + k_cat_A))
P_A = phi/(1 + phi)
```

With `a_X = k_cat_X/(k_off_X + k_cat_X)`, the probability that a bound
recombinase catalyses rather than dissociating, `P_A = a_A/(a_A + a_B)`.

`R` does not appear: both sites are entered from state 0 at the same rate
`k_on*R`, which cancels. Checked three ways. `sensitivity.phi_versus_R`
recomputes φ over five orders of magnitude in `R` and returns bit-for-bit
identical values; `sensitivity.symbolic_invariance_report` has SymPy return
`dphi/dR = 0`; the Gillespie simulation gives `P_A = 0.5` at `R` = 0.5, 5 and
500 nM.

### Equation 2, timescale to reach A or B

```
tau = (1 + (k_on/k_off_A + k_on/k_off_B)*R)
      / ((k_cat_A*k_on/k_off_A + k_cat_B*k_on/k_off_B)*R)
```

In seconds. `k_on/k_off` has units nM⁻¹, so the numerator is dimensionless and
the denominator s⁻¹. Every τ column in the workbook is in seconds.

### Approximation in Equation 2

Equation 2 is the mean first-passage time with `k_off_X + k_cat_X` replaced by
`k_off_X`, the `k_off >> k_cat` simplification. The unapproximated expression,
`model.tau_exact_mfpt`, is

```
T = (1/(2*k_on*R) + (1/2)*(1/(k_off_A+k_cat_A) + 1/(k_off_B+k_cat_B)))
    / ((w_A + w_B)/2),   w_X = k_cat_X/(k_off_X + k_cat_X)
```

and is tabulated beside Equation 2 in `Fig1D_tau`.

| `R` (nM) | Equation 2 | exact MFPT | difference |
|---|---|---|---|
| 0.1 | 87 750 s | 92 750 s | −5.39% |
| 1 | 9 000 s | 9 500 s | −5.26% |
| 10 | 1 125 s | 1 175 s | −4.26% |
| 50 | 425 s | 435 s | −2.30% |
| 240 | 286.5 s | 288.5 s | −0.72% |

Equation 2 understates the commitment time by at most 5% over this range, and
the gap narrows as `R` rises. The Gillespie simulation at `R` = 50 nM gives a
mean commitment time of 434.8 s, against 435.0 s exact and 425.0 s from
Equation 2.

τ enters only the `Fig1D_tau` sheet and the `exp(-T/tau)` undifferentiated
fraction in `SensC_undiff_curve`, where the master-equation result is tabulated
beside it.

### Equation 3, recombinase expression

```
dR/dt = beta_0 - gamma*R + sigma*R*eta(t)
```

Itô, integrated by Euler–Maruyama with multiplicative white noise:
`R_{n+1} = R_n + (beta_0 - gamma*R_n)*dt + sigma*R_n*sqrt(dt)*Z_n`, with `R`
clamped at 0 after each step.

Trajectories start at `R(0) = 0` and accumulate towards
`R_ss = beta_0/gamma = 50 nM`, which is not reached inside the window. The
closed-form `sigma = 0` solution `R(t) = R_ss*(1 - exp(-gamma*t))` gives
44.2337 nM at `t_end`; the Euler–Maruyama trajectory gives 44.2356 nM. A small
fraction of cells is therefore still uncommitted at `t_end` with no noise.

## Parameters

Defined once, in `src/parameters.py`, and not repeated elsewhere.

| symbol | value | units | source |
|---|---|---|---|
| `k_on` | 0.001 | nM⁻¹ s⁻¹ | literature estimate (crowding, non-facilitated diffusion) |
| `K_D` | 70 | nM | Ghosh et al. 2005; Singh et al. 2013 |
| `k_off_A` = `k_off_B` | 0.07 | s⁻¹ | derived, `= K_D * k_on` |
| `k_cat_A` = `k_cat_B` | 0.004 | s⁻¹ | Chao et al. 2021 FEBS J, Table 1 (3.69e-3, rounded) |
| `beta_0` | 0.0075 | nM s⁻¹ | burst-size derivation below |
| `gamma` | 0.00015 | s⁻¹ | 77 min doubling time |
| `sigma` | 0 and 0.064 | s⁻¹ᐟ² | the two trajectory conditions |
| `sigma` sweep | 0 to 0.075, step 0.005 | s⁻¹ᐟ² | 16 points |
| `dt` | 2 | s | integration step |
| `t_end` | 14400 | s (240 min) | — |
| `R(0)` | 0 | nM | expression starts from zero |
| `R_ss` | 50 | nM | derived, `= beta_0/gamma` |
| `seed` | 2025 | — | `numpy.random.default_rng` |

Recombinase abundance is in nM; for *E. coli* one molecule per cell is
approximately 1 nM (BioNumbers BNID 102068; Milo & Phillips, *Cell Biology by
the Numbers*). The arithmetic for a 1 fL cell gives 1.66 nM, and *E. coli*
volume ranges 0.44–1.79 µm³ (BNID 100004), so the rounding sits inside the
spread.

`beta_0 = 0.0075 nM/s` follows from burst statistics. With
`gamma = 0.00015 s⁻¹` the doubling time is 77 min. Burst frequency for active
genes in *E. coli* is about 5 mRNAs per cell cycle, and burst size spans 1 to
several hundred proteins per transcript, median 3 and mean 21 (Taniguchi et
al., *Science* 2010).

| burst size *b* | `beta_0` (nM/s) | `R_ss` (nM) |
|---|---|---|
| 3 (median) | 0.0032 | 22 |
| 7 | 0.0075 | 50 (used here) |
| 21 (mean) | 0.023 | 151 |
| 100 | 0.108 | 721 |

`parameters.unequal_rate_parameters()` holds a biased example,
`k_cat_A = 0.023` against `k_cat_B = 0.004`, giving φ = 4.575 and
`P_A` = 0.8206. The switch above has equal catalytic rates, so φ = 1 and the
two fates are equally likely. These are not the rates of the switch
characterised here.

## Repository map

| file | contents |
|---|---|
| `make_results.py` | writes `results/model_results.xlsx` |
| `src/parameters.py` | every parameter |
| `src/model.py` | Equations 1 and 2, the master equation, the batched propagator for time-varying `R` |
| `src/langevin.py` | Equation 3, seeded Euler–Maruyama |
| `src/gillespie.py` | stochastic cross-check of the master equation |
| `src/sensitivity.py` | the four sensitivity analyses |
| `src/thresholds.py` | reduction of Equation 2 and the division-time thresholds |
| `check_env.py` | version check against `requirements.txt` |
| `run_all.sh` | version check, then `make_results.py` |
| `requirements.txt` | pinned dependency versions |

## How each result is computed

Equations 1 and 2 are closed forms, evaluated directly.

`Fig1C_sweep` draws `K_D_A`, `K_D_B`, `k_cat_A`, `k_cat_B` independently and
log-uniformly (`K_D` 1 nM–1 µM, `k_cat` 1e-4–0.1 s⁻¹), 4000 samples, seeded.
`k_on` is held at 0.001 nM⁻¹ s⁻¹ and swept separately in `SensB_phi_grid`.
These ranges reach φ ≈ 1e-4 to 1e4, so the sampled points cover the inner four
decades of the analytic curve, which spans 1e-5 to 1e5.

`Fig1E_noise_sweep`: for each of the 16 σ values, 1000 independent Langevin
trajectories (seeded `SEED + i`), then the master equation integrated along
each cell's own `R(t)`. The reported value is the mean of `p_A(t_end)` over
cells.

`p_A + p_B` falls short of 1 at large σ. Those cells have not committed by
`t_end`, and the shortfall is reported as its own column. It rises
monotonically with σ, to 0.15% at σ = 0.075. Among cells that have committed,
`p_A/(p_A + p_B)` is 0.5 at every σ; the largest departure of `P_A` from `P_B`
on the grid is 2.8e-15.

Sensitivity analysis, four parts:

- `SensA_invariance`: φ against `R` over 1e-2 to 1e3 nM, and against a common
  factor on all four rate constants over 1e-2 to 1e2. Both flat. SymPy returns
  `dphi/dR = 0` and `phi(c·k) − phi(k) = 0`.
- `SensB_phi_grid`: φ over `k_on` ∈ [1e-4, 1e-2] × `K_D` ∈ [1, 1000], 41 × 41.
  φ = 1 throughout to floating-point precision while `k_cat_A = k_cat_B`.
- `SensC_undiff_curve` and `SensC_undiff_table`: at low `R` the switch is slow
  and cells accumulate undifferentiated.

  | `R` (nM) | τ | undifferentiated at 6 h | at 24 h |
  |---|---|---|---|
  | 0.05 | 48.7 h | 88.4% | 61.1% |
  | 0.1 | 24.4 h | 78.2% | 37.4% |
  | 0.3 | 8.2 h | 48.0% | 5.3% |
  | 1 | 2.5 h | 9.1% | 0.0% |
  | 50 | 0.12 h | 0.0% | 0.0% |

- `SensD_ratio`: the same low-`R` range, for the ratio among committed cells.
  It is 0.5 throughout, to 1e-12. Low `R` changes how many cells differentiate,
  not which state they adopt.

## Division-time thresholds

`src/thresholds.py`. Substituting `k_off_X = K_D_X * k_on` into Equation 2
gives `k_on/k_off_X = 1/K_D_X`, and for equal sites

```
tau = (K_D/2 + R) / (R * k_cat)        equivalently   tau * k_cat = 1 + K_D/(2R)
```

`k_on` cancels: `d(tau)/d(k_on) = 0`, confirmed by SymPy and by a numerical
check that a 100-fold change in `k_on` at fixed `K_D` leaves τ unchanged to
1e-12. The timescale is set by `K_D`, `k_cat` and `R`.

The switch outpaces division when

```
k_cat * t_div  >  1 + K_D/(2R)
```

Two dimensionless groups set the boundary: turnovers available per division,
`k_cat·t_div`, and the occupancy term `K_D/(2R)`. At the parameters used here
`K_D/2R = 0.7`, so 1.7 turnovers per division are required against 18.5
available. `Thresh_dimensionless` tabulates the boundary curve.

Solving `tau = t_div` for each parameter at `R = R_ss` = 50 nM:

| division time | `R*` (nM) | `k_cat*` (s⁻¹) | `K_D*` (nM) | turnovers available |
|---|---|---|---|---|
| 20 min | 9.21 | 1.42e-3 | 380 | 4.8 |
| 40 min | 4.07 | 7.08e-4 | 860 | 9.6 |
| 77 min (this model) | 2.00 | 3.68e-4 | 1748 | 18.5 |
| 4 h | 0.618 | 1.18e-4 | 5660 | 57.6 |
| 24 h | 0.102 | 1.97e-5 | 34460 | 345.6 |

The switch is slower than division below `R*`, below `k_cat*`, and above `K_D*`
(weaker binding). Each is checked by confirming `tau(R*) = t_div` to 1e-6 s.
Margins at the parameters used here run 5.4× to 492× in `R`, 2.8× to 203× in
`k_cat`, and 5.4× to 492× in `K_D`. The `R` and `K_D` margins are equal because
τ depends on them only through `K_D/R`. Slower growth lowers `R*`.

The exact mean first-passage time keeps a weak `k_on` dependence through the
search term `1/(2*k_on*R)`:

```
T = (K_D*k_on + 2*R*k_on + k_cat) / (2*R*k_cat*k_on)
k_on* = k_cat / (2*R*k_cat*t_div - K_D - 2*R)
```

which runs from 1.29e-5 nM⁻¹ s⁻¹ at 20 min to 1.16e-7 at 24 h, 78× to 8600×
below the `k_on` used here. Reported as `k_on_threshold_exact`.

Division times are defined in `thresholds.DIVISION_TIMES_S`: 20 min, 40 min,
77 min, 4 h, 24 h. Only the 77 min entry follows from the model, as
`ln(2)/gamma` with `gamma` = 1.5e-4 s⁻¹. The others are round values bracketing
laboratory culture to host and soil colonisation, not taken from a specific
measurement; any value quoted in the manuscript needs its own citation.
`Thresh_vs_t_div` gives continuous curves for other division times.

## Fluctuation magnitude

A single noisy trajectory is one realisation, so `Fig1E_fluctuations` tabulates
the distribution of `R` across cells at `t_end`, for every σ on the sweep grid
plus σ = 0.064.

The standard deviation is not a usable statistic here. The stationary density
of Equation 3 under multiplicative noise is inverse-gamma with shape
`1 + 2*gamma/sigma^2`, whose variance is finite only for
`sigma < sqrt(2*gamma)` = 0.01732 s⁻¹ᐟ². Above that the SD does not converge as
cells are added or the run is lengthened: it jumps around non-monotonically
(667 nM at σ = 0.025, 43 nM at σ = 0.070) while the median and percentiles fall
smoothly. Use percentiles.

At σ = 0.064, median `R` = 4.85 nM, interquartile range 2.43–11.41 nM, 5th to
95th percentile 1.13–57.15 nM. At σ = 0 every cell sits at 44.2356 nM.

## Validation

`make_results.py` recomputes 55 checks and writes them to the `validation`
sheet with expected value, computed value, absolute difference, tolerance and
result. The script exits non-zero if any check fails.

Each check is an analytic value of the model equations, a Monte Carlo estimate
of one, or a property of the implementation. Properties are expressed as
violations, so an expected value of 0 means it does not occur.

| group | checks |
|---|---|
| Equation 1 | φ and `P_A` at equal and unequal catalytic rates; spread of φ over `R` ∈ [1e-2, 1e3]; spread under a common rescaling of all four rates; spread over the `k_on` × `K_D` grid |
| Equation 2 | τ at `R` = 1 and 100 nM, the `R → ∞` limit, τ at unequal catalytic rates |
| master equation | probability conservation; `p_A`, `p_B` and the uncommitted fraction at `R` = 50 nM, `t_end` |
| Gillespie | `P_A` at equal and unequal catalytic rates over 100 000 cells; `R`-independence at `R` = 0.5, 5 and 500 nM; mean commitment time against the exact MFPT |
| propagator | batched `expm` against `scipy.linalg.expm`; constant-`R` propagation against `solve_ivp`; conservation and non-negativity along 100 noisy trajectories |
| Equation 3 | `R(t_end)` against the closed form; the σ = 0 trace rises monotonically and stays below `R_ss`; timestep grid; `R` never negative; the same seed reproduces a trajectory |
| σ grid | 16 points, spacing 0.005, σ = 0.064 off-grid |
| noise sweep | `P_A` at σ = 0; `P_A == P_B` at every σ; committed-cell ratio 0.5 at every σ; undifferentiated fraction rises monotonically with σ |
| units | `k_on/k_off = 1/K_D`; `t_end` is 240 min |
| thresholds | reduced τ against Equation 2; τ unchanged by a 100× change in `k_on` at fixed `K_D`; `tau(R*) = t_div` at every division time; `R*` and the operating-point margin |
| fluctuations | the σ below which the stationary variance is finite; zero spread at σ = 0 |
| sensitivity | all ten percentages above; the single-exponential shortcut against the master equation |

Values from a clean run: φ spread over `R` exactly 0; `R(t_end)` 44.2356 nM
against a closed form of 44.2337; Gillespie `P_A` 0.49797 over 100 000 cells;
mean commitment time 434.8 s against 435.0 s exact; `P_A` 0.500000 at σ = 0 and
0.499244 at σ = 0.075; largest `|P_A − P_B|` on the σ grid 2.8e-15.

## Numerical methods

- Constant `R`: `scipy.integrate.solve_ivp` with LSODA, `rtol = 1e-10`,
  `atol = 1e-12`.
- Time-varying `R`: `R(t)` is piecewise constant on the `dt = 2 s` grid, so the
  propagator over each step is `expm(Q(R_n)*dt)`. Computed for all cells at
  once by scaling-and-squaring with a 12-term Taylor series
  (`model._expm_batch`), checked against `scipy.linalg.expm` to 2e-13. Cells
  whose uncommitted probability falls below 1e-14 are dropped from the batch.
- Gillespie: direct SSA, vectorised across cells with exponential waiting
  times. Used as a cross-check, not as a source of reported values.
- Langevin: Euler–Maruyama, `dt = 2 s`, 7200 steps, `R` clamped at 0.

## Licence

MIT, see `LICENSE`. Please cite the manuscript.
