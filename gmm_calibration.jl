# =============================================================================
# GMM Calibration Scaffolding (additive layer on Merged_Model.jl)
# =============================================================================
#
# Does NOT run optimization — provides moment computation, data targets,
# and the GMM objective for advisor review before a full estimation.
#
# Fixed (not estimated): beta, sigma, alpha, r, nu, lambda, grids, ability
#   support choices (use_pareto_z, z_min, fixed_z, rho_z, sigma_z, n_z, n_a, ...).
#
# Estimated (11): b, s, d, eta, kappa_v, kappa_e, cf, A, xi, alpha_z, delta.
#   k* is endogenous (from MPK = user cost), NOT estimated.
# =============================================================================

include("Merged_Model.jl")

using LinearAlgebra, Printf, Statistics

# ---- Parameter bookkeeping ---------------------------------------------------

const ESTIMATED_PARAM_NAMES = (
    :b, :s, :d, :eta, :kappa_v, :kappa_e, :cf, :A, :xi, :alpha_z, :delta,
)

const MOMENT_NAMES = (
    :entrepreneur_rate,              # self-employment share of population
    :unemployment_rate,              # u / (u + w)
    :mean_firm_duration,            # avg completed entrepreneur spell (periods)
    :mean_unemployment_duration,     # avg completed unemployment spell (periods)
    :mean_job_tenure,                # avg completed worker spell (periods)
    :unemp_to_wage_income_ratio,     # mean UI income / mean worker wage income
    :worker_to_entrepreneur_income_ratio,  # mean worker / mean entrepreneur income
    :wealth_to_income_ratio,         # mean assets / mean total labor income
    :mean_firm_size,                 # mean firm size (workers) among entrepreneurs
    :std_firm_size,                 # std dev of firm size among entrepreneur-period obs
)

"""Default estimated-parameter vector (same order as ESTIMATED_PARAM_NAMES)."""
function default_estimated_vector()
    p = make_params()
    return [getfield(p, nm) for nm in ESTIMATED_PARAM_NAMES]
end

"""Build Params from an estimated-parameter vector + optional fixed overrides."""
function params_from_estimated(θ::AbstractVector; kwargs...)
    length(θ) == length(ESTIMATED_PARAM_NAMES) ||
        error("expected $(length(ESTIMATED_PARAM_NAMES)) parameters, got $(length(θ))")
    est = NamedTuple{ESTIMATED_PARAM_NAMES}(Tuple(θ))
    return make_params(;
        b=est.b, s=est.s, d=est.d, eta=est.eta,
        kappa_v=est.kappa_v, kappa_e=est.kappa_e, cf=est.cf,
        A=est.A, xi=est.xi, alpha_z=est.alpha_z, delta=est.delta,
        kwargs...)
end

# ---- Data targets (PLACEHOLDER — replace with country-specific data) ---------

"""
    data_target_moments()

Returns a NamedTuple of target moments. Every value is a PLACEHOLDER drawn from
broad developing-economy benchmarks / typical macro-calibration ranges.
Replace with your estimation sample before running GMM.
"""
function data_target_moments()
    return (
        entrepreneur_rate              = 0.06,   # PLACEHOLDER: Gindling & Newhouse ~5-7% employer/self-employed in LICs
        unemployment_rate              = 0.08,   # PLACEHOLDER: open unemployment, developing-economy ballpark
        mean_firm_duration             = 12.0,   # PLACEHOLDER: ~12 years avg firm life (1/exit ≈ 1/d in years)
        mean_unemployment_duration     = 0.75,   # PLACEHOLDER: ~9 months (model periods = years)
        mean_job_tenure                = 4.0,    # PLACEHOLDER: mean worker job spell, developing economies (years)
        unemp_to_wage_income_ratio     = 0.40,   # PLACEHOLDER: UI replacement rate b/w, common in calibrations
        worker_to_entrepreneur_income_ratio = 0.85,  # PLACEHOLDER: workers earn less than entrepreneurs on average
        wealth_to_income_ratio         = 3.0,    # PLACEHOLDER: aggregate wealth/income, middle-income range
        mean_firm_size                 = 3.0,    # PLACEHOLDER: small firms in developing economies (workers)
        std_firm_size                  = 4.0,    # PLACEHOLDER: wide firm-size dispersion
    )
end

function moment_vector(m::NamedTuple)
    return [m[k] for k in MOMENT_NAMES]
end

function target_vector(targets=data_target_moments())
    return [targets[k] for k in MOMENT_NAMES]
end

# ---- Extended simulation for spell / income moments --------------------------

"""
    entrepreneur_income(z_val, ell_val, a_val, theta, p)

Period labor/operating income for an entrepreneur (net of wages paid and
capital user cost, minus fixed operating cost cf). Matches the household block.
"""
function entrepreneur_income(z_val, ell_val, a_val, theta, p)
    keff = k_eff(z_val, ell_val, a_val, p)
    w = wage_sched(z_val, ell_val, theta, p)
    return f_output(z_val, ell_val, keff, p) - w * ell_val - ucost(p) * keff - p.cf
end

"""
    simulate_for_gmm(theta, hh, p; N, T, burn, seed)

Extended simulation that tracks completed spells, period incomes, and firm-size
cross-sections needed for GMM moments. Uses the SAME policies and the SAME RNG
call sequence as `simulate_population`, so population shares are identical.

SPELL HANDLING (durations + firm-level size):
  A "spell" is a contiguous run in one labor-market state; a "firm" is a
  contiguous entrepreneur spell (firm IDs = these runs; firms start at ell=0).
  Only spells that BEGIN after burn-in are recorded, so every recorded spell is
  fully observed in stationarity — no left-censoring and no contamination from
  the non-stationary initial conditions (everyone starts unemployed at a=0).
  Spells still open at t=T (right-censored) are dropped and counted separately
  (`cens_*`) to confirm the bias is negligible given mean durations << T-burn.
"""
function simulate_for_gmm(theta, hh, p; N=20000, T=600, burn=300, seed=1)
    rng = MersenneTwister(seed)
    ft = f_theta(theta, p)
    delta = p.s + p.d - p.s * p.d
    qt = q_theta(theta, p)
    cum = cumsum(p.Pi_z, dims=2)
    cum_pi = cumsum(p.pi_z)

    a = fill(0.0, N)
    z = [searchsortedfirst(cum_pi, rand(rng)) for _ in 1:N]
    state = fill(1, N)
    ell = fill(0.0, N)

    # spell bookkeeping: length so far and the period the current spell began
    spell_len = zeros(Int, N)
    spell_start = fill(1, N)             # initial spell begins at t=1 (<= burn)

    # per-firm size accumulators (one firm = one contiguous entrepreneur spell)
    firm_ell_sum = zeros(N)              # sum of operating size over the spell
    firm_ell_cnt = zeros(Int, N)         # periods operated
    firm_ell_last = zeros(N)             # size in the final operating period

    # completed-spell samples (spells that STARTED after burn-in only)
    u_spells = Float64[]; w_spells = Float64[]; e_spells = Float64[]
    firm_avg_size = Float64[]            # one obs per firm: mean size over its life
    firm_end_size = Float64[]            # one obs per firm: size at end of life

    # period (flow) cross-section observations, post burn-in
    inc_u = Float64[]; inc_w = Float64[]; inc_e = Float64[]   # labor/business income
    wealth_obs = Float64[]               # assets a
    tot_income_obs = Float64[]           # r*a (capital income) + labor/business income
    cap_income_obs = Float64[]           # r*a only (diagnostic)
    firm_size_obs = Float64[]            # period-based size (feeds mean_firm_size)

    nearest_a(x) = clamp(searchsortedfirst(p.a_grid, x), 1, p.n_a)
    nearest_ell(x) = clamp(searchsortedfirst(hh.ell_grid_e, x), 1, length(hh.ell_grid_e))

    record_spell!(st, len) = st == 1 ? push!(u_spells, float(len)) :
                             st == 2 ? push!(w_spells, float(len)) :
                                       push!(e_spells, float(len))

    # Close the current spell of person i (in old_state) at period t_now and
    # start a fresh spell. Record only spells that began strictly after burn-in.
    function close_spell!(i, t_now, old_state)
        if spell_start[i] > burn && spell_len[i] > 0
            record_spell!(old_state, spell_len[i])
            if old_state == 3 && firm_ell_cnt[i] > 0
                push!(firm_avg_size, firm_ell_sum[i] / firm_ell_cnt[i])
                push!(firm_end_size, firm_ell_last[i])
            end
        end
        spell_len[i] = 0
        spell_start[i] = t_now
        firm_ell_sum[i] = 0.0; firm_ell_cnt[i] = 0; firm_ell_last[i] = 0.0
    end

    share_u = 0.0; share_w = 0.0; share_e = 0.0; nrec = 0

    for t in 1:T
        for i in 1:N
            ia = nearest_a(a[i]); iz = z[i]
            spell_len[i] += 1

            # voluntary entrepreneur exit (same timing as simulate_population):
            # exits at the start of the period, becomes unemployed, does NOT
            # operate this period -> close the firm spell here.
            if state[i] == 3 && hh.exitE[ia, iz, nearest_ell(ell[i])]
                close_spell!(i, t, 3)
                state[i] = 1; ell[i] = 0.0
            end

            state_before = state[i]

            if state[i] == 1
                if hh.occ[ia, iz] == 2
                    a[i] = hh.apE0[ia, iz]; ell[i] = 0.0; state[i] = 3
                else
                    a[i] = hh.apU[ia, iz]
                    state[i] = rand(rng) < ft ? 2 : 1
                end
            elseif state[i] == 2
                if hh.quitW[ia, iz]
                    a[i] = hh.apE0[ia, iz]; ell[i] = 0.0; state[i] = 3
                else
                    a[i] = hh.apW[ia, iz]
                    state[i] = rand(rng) < delta ? 1 : 2
                end
            else
                iel = nearest_ell(ell[i])
                # operate the firm this period at its current size ell[i]
                firm_ell_sum[i] += ell[i]; firm_ell_cnt[i] += 1; firm_ell_last[i] = ell[i]
                a[i] = hh.apE[ia, iz, iel]
                v = interp_ell(p.ell_grid, hh.v_pol[iz, :], ell[i])
                ellnew = (1 - p.s) * ell[i] + qt * v
                if rand(rng) < p.d
                    state[i] = 1; ell[i] = 0.0
                else
                    state[i] = 3; ell[i] = ellnew
                end
            end

            if state[i] != state_before
                close_spell!(i, t, state_before)
            end

            if !p.fixed_z
                r = rand(rng); z[i] = searchsortedfirst(cum[iz, :], r)
            end

            if t > burn
                push!(wealth_obs, a[i])
                cap_inc = p.r * a[i]
                if state[i] == 1
                    lab = p.b; push!(inc_u, lab)
                elseif state[i] == 2
                    lab = hh.wage_z[iz]; push!(inc_w, lab)
                else
                    z_val = p.z_grid[iz]
                    lab = entrepreneur_income(z_val, ell[i], a[i], theta, p)
                    push!(inc_e, lab); push!(firm_size_obs, ell[i])
                end
                push!(tot_income_obs, cap_inc + lab)
                push!(cap_income_obs, cap_inc)
            end
        end

        if t > burn
            share_u += count(==(1), state) / N
            share_w += count(==(2), state) / N
            share_e += count(==(3), state) / N
            nrec += 1
        end
    end

    # right-censored spells: open at T and started after burn-in (dropped above)
    cens_u = 0; cens_w = 0; cens_e = 0
    for i in 1:N
        if spell_start[i] > burn && spell_len[i] > 0
            state[i] == 1 ? (cens_u += 1) :
            state[i] == 2 ? (cens_w += 1) : (cens_e += 1)
        end
    end

    share_u /= nrec; share_w /= nrec; share_e /= nrec
    urate = share_u / (share_u + share_w)

    return (
        share_u=share_u, share_w=share_w, share_e=share_e, urate=urate,
        u_spells=u_spells, w_spells=w_spells, e_spells=e_spells,
        firm_avg_size=firm_avg_size, firm_end_size=firm_end_size,
        inc_u=inc_u, inc_w=inc_w, inc_e=inc_e,
        wealth_obs=wealth_obs, tot_income_obs=tot_income_obs,
        cap_income_obs=cap_income_obs, firm_size_obs=firm_size_obs,
        cens_u=cens_u, cens_w=cens_w, cens_e=cens_e,
    )
end

# ---- Model moments -----------------------------------------------------------

"""
    model_moments(sol; gmm_sim=nothing)

Compute all advisor target moments from a solved model.

Uses `sol.sim` for basic shares when available; runs `simulate_for_gmm` (or uses
a precomputed `gmm_sim` result) for spell, income, and firm-size dispersion moments.

Moment definitions:
  entrepreneur_rate       — share of population in state entrepreneur (simulation).
  unemployment_rate       — unemployed / (unemployed + workers); excludes entrepreneurs.
  mean_firm_duration      — mean length of COMPLETED entrepreneur spells (periods),
                            among spells that started post burn-in. Includes exogenous
                            exit (d) and voluntary exit; 1/d is a poor proxy because
                            voluntary exit dominates.
  mean_unemployment_duration — mean length of COMPLETED unemployment spells (post-burn
                            starts) until job find or firm start.
  mean_job_tenure         — mean length of COMPLETED worker spells (post-burn starts)
                            until separation or quit-to-entrepreneurship.
  unemp_to_wage_income_ratio — mean(period UI income) / mean(period worker wage),
                            post burn-in; UI income = b each unemployed period.
  worker_to_entrepreneur_income_ratio — mean worker wage / mean entrepreneur net
                            operating income (after cf, wages paid, capital cost).
                            Both are LABOR/BUSINESS income, excluding capital income.
  wealth_to_income_ratio  — mean(assets) / mean(TOTAL period income), aggregate ratio.
                            Total period income (flow) = capital income r*a +
                            labor/business income (b, w, or net profit). Uses r*a, NOT
                            (1+r)*a: the principal a is a stock, not income.
  mean_firm_size          — mean firm size (workers) over entrepreneur-PERIOD obs
                            (≈ stationary cross-section of operating firms).
  std_firm_size           — std dev of firm size across FIRMS (one obs per completed
                            firm = its average size over the firm's life). See the
                            diagnostics for the period-based and end-of-life variants.
"""
function model_moments(sol; gmm_sim=nothing, sim_kwargs...)
    p = sol.p; theta = sol.theta; hh = sol.hh
    gsim = gmm_sim === nothing ?
        simulate_for_gmm(theta, hh, p; sim_kwargs...) : gmm_sim

    # Easy moments (also in sol.sim, but re-use gsim for consistency)
    entrepreneur_rate = gsim.share_e
    unemployment_rate = gsim.urate

    mean_firm_duration = isempty(gsim.e_spells) ? NaN : mean(gsim.e_spells)
    mean_unemployment_duration = isempty(gsim.u_spells) ? NaN : mean(gsim.u_spells)
    mean_job_tenure = isempty(gsim.w_spells) ? NaN : mean(gsim.w_spells)

    mean_w = isempty(gsim.inc_w) ? NaN : mean(gsim.inc_w)
    mean_ui = isempty(gsim.inc_u) ? NaN : mean(gsim.inc_u)
    mean_ent_inc = isempty(gsim.inc_e) ? NaN : mean(gsim.inc_e)

    unemp_to_wage_income_ratio = mean_w > 0 ? mean_ui / mean_w : NaN
    worker_to_entrepreneur_income_ratio =
        mean_ent_inc > 0 ? mean_w / mean_ent_inc : NaN

    # Wealth-to-income ratio: aggregate mean assets / mean TOTAL period income,
    # where total income = capital income (r*a) + labor/business income. This is
    # the standard wealth/income ratio (a stock over an annual flow). Using r*a
    # rather than (1+r)*a is deliberate — (1+r)*a folds the asset principal into
    # "income" and would mechanically push any ratio toward ~1/(1+r).
    mean_wealth = isempty(gsim.wealth_obs) ? NaN : mean(gsim.wealth_obs)
    mean_tot_income = isempty(gsim.tot_income_obs) ? NaN : mean(gsim.tot_income_obs)
    wealth_to_income_ratio = mean_tot_income > 0 ? mean_wealth / mean_tot_income : NaN

    mean_firm_size = isempty(gsim.firm_size_obs) ? NaN : mean(gsim.firm_size_obs)
    # firm-level dispersion: one observation per completed firm = average size
    # over that firm's life (firm IDs = contiguous entrepreneur spells).
    std_firm_size = length(gsim.firm_avg_size) > 1 ? std(gsim.firm_avg_size) : NaN

    # ---- diagnostics (not part of the GMM moment vector) --------------------
    all_labor_income = vcat(gsim.inc_u, gsim.inc_w, gsim.inc_e)
    mean_labor_income = isempty(all_labor_income) ? NaN : mean(all_labor_income)
    mean_cap_income = isempty(gsim.cap_income_obs) ? NaN : mean(gsim.cap_income_obs)
    # OLD (buggy) definition: assets / labor income only — kept for comparison
    wealth_to_labor_income_ratio = mean_labor_income > 0 ? mean_wealth / mean_labor_income : NaN
    # alternative firm-size dispersions
    std_firm_size_period = length(gsim.firm_size_obs) > 1 ? std(gsim.firm_size_obs) : NaN
    std_firm_size_end = length(gsim.firm_end_size) > 1 ? std(gsim.firm_end_size) : NaN
    mean_firm_size_perfirm = isempty(gsim.firm_avg_size) ? NaN : mean(gsim.firm_avg_size)

    return (
        entrepreneur_rate=entrepreneur_rate,
        unemployment_rate=unemployment_rate,
        mean_firm_duration=mean_firm_duration,
        mean_unemployment_duration=mean_unemployment_duration,
        mean_job_tenure=mean_job_tenure,
        unemp_to_wage_income_ratio=unemp_to_wage_income_ratio,
        worker_to_entrepreneur_income_ratio=worker_to_entrepreneur_income_ratio,
        wealth_to_income_ratio=wealth_to_income_ratio,
        mean_firm_size=mean_firm_size,
        std_firm_size=std_firm_size,
        # diagnostics (not in GMM vector)
        _implied_firm_duration_1_over_d=1.0 / p.d,
        _n_u_spells=length(gsim.u_spells),
        _n_w_spells=length(gsim.w_spells),
        _n_e_spells=length(gsim.e_spells),
        _n_firms=length(gsim.firm_avg_size),
        _cens_u=gsim.cens_u, _cens_w=gsim.cens_w, _cens_e=gsim.cens_e,
        _mean_wealth=mean_wealth,
        _mean_labor_income=mean_labor_income,
        _mean_cap_income=mean_cap_income,
        _mean_tot_income=mean_tot_income,
        _wealth_to_labor_income_ratio=wealth_to_labor_income_ratio,
        _std_firm_size_period=std_firm_size_period,
        _std_firm_size_end=std_firm_size_end,
        _mean_firm_size_perfirm=mean_firm_size_perfirm,
    )
end

# ---- Quiet solve (for repeated GMM evaluations) ------------------------------

function solve_model_quiet(p)
    theta, E_firm, v_pol = solve_labor_market(p)
    hh = solve_household(theta, E_firm, v_pol, p; verbose=false)
    sim = simulate_population(theta, hh, p)
    return (theta=theta, E_firm=E_firm, v_pol=v_pol, hh=hh, sim=sim, p=p)
end

# ---- GMM objective -----------------------------------------------------------

"""
    gmm_objective(θ; targets=data_target_moments(), W=I, sim_kwargs...)

Set estimated parameters θ, solve the model, simulate, compute moments, and
return (objective, moments, sol).

Weighting matrix W defaults to identity (efficient GMM would use optimal W
from a first-stage or continuous-updating estimator — see notes in test script).
"""
function gmm_objective(θ::AbstractVector;
    targets=data_target_moments(),
    W=I,
    sim_kwargs...)
    p = params_from_estimated(θ)
    sol = solve_model_quiet(p)
    gsim = simulate_for_gmm(sol.theta, sol.hh, p; sim_kwargs...)
    moms = model_moments(sol; gmm_sim=gsim)
    m = moment_vector(moms)
    μ = target_vector(targets)
    diff = m - μ
    obj = real(diff' * W * diff)
    return (objective=obj, moments=moms, sol=sol, gmm_sim=gsim, diff=diff)
end

# ---- Reporting helpers -------------------------------------------------------

const MOMENT_DIFFICULTY = Dict(
    :entrepreneur_rate => ("easy", "Population share from simulation."),
    :unemployment_rate => ("easy", "u/(u+w) from simulation."),
    :mean_firm_duration => ("moderate", "Completed entrepreneur spells (post-burn starts); includes voluntary exit, not just 1/d."),
    :mean_unemployment_duration => ("moderate", "Completed unemployment spells (post-burn starts) until job find or firm start."),
    :mean_job_tenure => ("moderate", "Completed worker spells (post-burn starts) until separation or quit."),
    :unemp_to_wage_income_ratio => ("moderate", "Period UI (=b) vs worker wage means; excludes capital income."),
    :worker_to_entrepreneur_income_ratio => ("moderate", "Mean worker wage vs mean entrepreneur net operating income (labor/business income only)."),
    :wealth_to_income_ratio => ("hard", "Mean assets / mean total period income (r*a + labor/business). Sensitive to a_max and the income definition; see diagnostics."),
    :mean_firm_size => ("easy", "Mean ell over entrepreneur-period obs (stationary cross-section of operating firms)."),
    :std_firm_size => ("moderate", "Std across FIRMS (one obs per completed firm = avg size over its life). Period-based & end-of-life variants in diagnostics."),
)

function print_moment_comparison(moms, targets=data_target_moments())
    println("\n" * "="^78)
    println("MODEL MOMENTS vs DATA TARGETS (PLACEHOLDER)")
    println("="^78)
    println(@sprintf("  %-36s %12s %12s %12s %8s", "Moment", "Model", "Target", "Gap", "Level"))
    println("  " * "-"^74)
    mv = moment_vector(moms)
    tv = target_vector(targets)
    for (i, nm) in enumerate(MOMENT_NAMES)
        level, _ = MOMENT_DIFFICULTY[nm]
        gap = mv[i] - tv[i]
        println(@sprintf("  %-36s %12.4f %12.4f %12.4f %8s",
            String(nm), mv[i], tv[i], gap, level))
    end
end

function print_estimation_readiness()
    println("\n" * "="^78)
    println("WHAT REMAINS FOR FULL GMM ESTIMATION")
    println("="^78)
    println("""
  1. Replace PLACEHOLDER targets in data_target_moments() with country data.
  2. Optimizer: recommend Derivative-free first (Nelder-Mead / CMA-ES in Optim.jl
     or BlackBoxOptim) since solving is noisy and ~8 min/eval; gradient-based
     methods need smooth interpolation or larger N.
  3. Weighting matrix: start with identity (done); then two-step GMM —
     (a) estimate with W=I, (b) compute moment covariance from bootstrap or
     simulation runs at θ̂, set W = Σ⁻¹.
  4. Runtime: ~8 min per objective evaluation × ~11 parameters × hundreds of
     evals → days to weeks. Use parallel workers (pmap over θ batches) or
     reduce N,T for exploration, then refine at final θ.
  5. Identification: check Jacobian of moments w.r.t. θ (finite differences,
     11 extra solves) before committing to full search.
  6. Remaining moment caveats (computation now fixed; these are model/sim issues):
     - wealth/income: ratio stays ~20 (not ~3) because mean assets (~36) are large
       vs flow income and a_max=50 likely BINDS. Raise a_max and re-check, or
       revisit the b/r/beta gap and the collateral saving motive.
     - job tenure: ~5% of post-burn worker spells are right-censored at T -> tenure
       biased slightly DOWN. Raise T (e.g. 1000) if precision matters.
     - firm-size: choose firm-level (one-per-firm) vs period-based (cross-section)
       to MATCH how the data moment is built; both are reported.
""")
end
