# =============================================================================
# DMP Model with Multi-Worker Firms + Bewley-Huggett-Aiyagari Asset Market
# =============================================================================
#
# This EXTENDS the baseline DMP model (DMP.jl) by giving workers a savings
# decision and a borrowing constraint, following Bewley-Huggett-Aiyagari (BHA).
#
# WHAT'S NEW relative to DMP.jl:
#   - Workers hold financial assets a and choose how much to save each period
#   - CRRA utility u(c) = c^(1-sigma)/(1-sigma)
#   - A borrowing constraint a' >= a_min (workers are borrowing constrained)
#   - Worker value functions now carry assets:
#        W(a, z_firm, ell_firm) = employed worker
#        U(a)                   = unemployed worker
#   - Savings choice solved by CONTINUOUS optimization over a' with the
#     continuation value interpolated over the asset grid (smooth policies)
#   - A stationary wealth distribution over (a, employment status, firm state),
#     simulated with assets tracked as a CONTINUOUS variable
#   - Aggregate capital K = average assets in the economy
#
# WAGE BARGAINING:
#   Workers bargain with threat point b (they threaten to STRIKE, not quit),
#   NOT their unemployment value U. This keeps the wage independent of a
#   worker's own assets, which avoids a curse of dimensionality. The baseline
#   nash_wage already uses b as the threat point, so it is unchanged here.
#
# NOT YET INCLUDED (next step):
#   - Worker's own entrepreneurial ability z_self
#   - Occupational choice: unemployed agent picks max(W-search, E-entrepreneur)
#   These plug into U and the worker problem; see SECTION 13 note.
#
# =============================================================================

using LinearAlgebra, Statistics, Printf, Optim, Random, Plots

# =============================================================================
# SECTION 1: Parameters
# =============================================================================

struct Params
    # --- Preferences ---
    beta    :: Float64      # discount factor
    sigma   :: Float64      # CRRA risk aversion
    b       :: Float64      # unemployment benefit

    # --- Production ---
    alpha   :: Float64      # capital share in f(z,ell,k) = z * k^alpha * ell^(1-alpha)
    k       :: Float64      # capital per firm (fixed/normalized)
    r       :: Float64      # rate of return on worker savings / capital rental

    # --- Labor market frictions ---
    s       :: Float64      # exogenous separation rate
    d       :: Float64      # exogenous firm destruction rate
    eta     :: Float64      # worker Nash bargaining power
    kappa_v :: Float64      # per-vacancy posting cost
    kappa_e :: Float64      # firm entry cost (free entry pins down theta)

    # --- Matching function: mu(u,v) = A * u^xi * v^(1-xi) ---
    A       :: Float64      # matching efficiency
    xi      :: Float64      # matching elasticity w.r.t. unemployment

    # --- Productivity grid (Tauchen discretization of AR(1) log z) ---
    n_z      :: Int
    z_grid   :: Vector{Float64}
    Pi_z     :: Matrix{Float64}
    pi_z     :: Vector{Float64}

    # --- Firm size grid ---
    n_ell    :: Int
    ell_grid :: Vector{Float64}

    # --- Asset grid (workers' savings) ---
    n_a      :: Int
    a_min    :: Float64
    a_max    :: Float64
    a_grid   :: Vector{Float64}
end

function make_params(;
    beta    = 0.96,
    sigma   = 2.00,
    b       = 0.40,
    alpha   = 0.33,
    k       = 1.00,
    r       = 0.03,
    s       = 0.03,
    d       = 0.02,
    eta     = 0.50,
    kappa_v = 0.50,
    kappa_e = 1.00,
    A       = 0.70,
    xi      = 0.50,
    n_z     = 7,
    rho_z   = 0.90,
    sigma_z = 0.10,
    n_ell   = 30,
    ell_max = 15.0,
    n_a     = 80,
    a_min   = 0.00,
    a_max   = 40.0,
)
    z_grid, Pi_z, pi_z = tauchen(n_z, rho_z, sigma_z)
    z_grid = exp.(z_grid)
    ell_grid = collect(range(0.1, ell_max, length=n_ell))
    # curved asset grid: denser near the borrowing constraint
    a_grid = a_min .+ (a_max - a_min) .* (range(0, 1, length=n_a)).^2
    return Params(beta, sigma, b, alpha, k, r, s, d, eta, kappa_v, kappa_e,
                  A, xi, n_z, z_grid, Pi_z, pi_z, n_ell, ell_grid,
                  n_a, a_min, a_max, a_grid)
end


# =============================================================================
# SECTION 2: Tauchen (1986) Discretization
# =============================================================================

function tauchen(n::Int, rho::Float64, sigma::Float64; m::Float64=3.0)
    sigma_unc = sigma / sqrt(1.0 - rho^2)
    z_max  = m * sigma_unc
    z_min  = -z_max
    z_grid = collect(range(z_min, z_max, length=n))
    dz     = z_grid[2] - z_grid[1]

    Pi = zeros(n, n)
    for i in 1:n
        for j in 1:n
            lo    = z_grid[j] - dz/2
            hi    = z_grid[j] + dz/2
            mu_ij = rho * z_grid[i]
            if j == 1
                Pi[i, j] = normal_cdf((hi - mu_ij) / sigma)
            elseif j == n
                Pi[i, j] = 1.0 - normal_cdf((lo - mu_ij) / sigma)
            else
                Pi[i, j] = normal_cdf((hi - mu_ij) / sigma) -
                            normal_cdf((lo - mu_ij) / sigma)
            end
        end
        Pi[i, :] ./= sum(Pi[i, :])
    end

    pi = fill(1.0/n, n)
    for _ in 1:2000
        pi_new = Pi' * pi
        maximum(abs.(pi_new - pi)) < 1e-12 && break
        pi = pi_new
    end
    return z_grid, Pi, pi
end

# Self-contained standard normal CDF (Abramowitz & Stegun 7.1.26 erf)
function erf_as(x::Float64)
    s = sign(x); x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * exp(-x * x)
    return s * y
end
normal_cdf(x::Float64) = 0.5 * (1.0 + erf_as(x / sqrt(2.0)))


# =============================================================================
# SECTION 3: Matching Functions
# =============================================================================

f_theta(theta::Float64, p::Params) = p.A * theta^(1.0 - p.xi)
q_theta(theta::Float64, p::Params) = p.A * theta^(-p.xi)


# =============================================================================
# SECTION 4: Production, MPL, Profits
# =============================================================================

f_prod(z::Float64, ell::Float64, p::Params) =
    z * p.k^p.alpha * ell^(1.0 - p.alpha)

MPL(z::Float64, ell::Float64, p::Params) =
    (1.0 - p.alpha) * z * p.k^p.alpha * ell^(-p.alpha)

pi_firm(z::Float64, ell::Float64, w::Float64, p::Params) =
    f_prod(z, ell, p) - w * ell - p.r * p.k


# =============================================================================
# SECTION 5: CRRA Utility and Nash Wage
# =============================================================================

function util(c::Float64, p::Params)
    c <= 0.0 && return -1e10
    if abs(p.sigma - 1.0) < 1e-8
        return log(c)
    else
        return (c^(1.0 - p.sigma) - 1.0) / (1.0 - p.sigma)
    end
end

function nash_wage(z::Float64, ell::Float64, theta::Float64, p::Params)
    mpl = MPL(z, ell, p)
    return p.eta * (mpl + p.kappa_v * theta) + (1.0 - p.eta) * p.b
end


# =============================================================================
# SECTION 6: Interpolation Helpers
# =============================================================================

# Linear interpolation over the ell grid (accepts views)
function interp_ell(ell_grid::Vector{Float64}, V::AbstractVector, ell::Float64)
    ell = clamp(ell, ell_grid[1], ell_grid[end])
    i   = searchsortedfirst(ell_grid, ell)
    i   = clamp(i, 2, length(ell_grid))
    t   = (ell - ell_grid[i-1]) / (ell_grid[i] - ell_grid[i-1])
    return (1.0 - t) * V[i-1] + t * V[i]
end

# Linear interpolation over the asset grid (accepts views)
function interp_a(V::AbstractVector, p::Params, a::Float64)
    a = clamp(a, p.a_grid[1], p.a_grid[end])
    i = searchsortedfirst(p.a_grid, a)
    i = clamp(i, 2, p.n_a)
    t = (a - p.a_grid[i-1]) / (p.a_grid[i] - p.a_grid[i-1])
    return (1.0 - t) * V[i-1] + t * V[i]
end

# Bilinear interpolation of W over (a, ell) at fixed productivity index iz2
function interp_W(W::Array{Float64,3}, p::Params,
                  a::Float64, iz2::Int, ell::Float64)
    a = clamp(a, p.a_grid[1], p.a_grid[end])
    ia = searchsortedfirst(p.a_grid, a); ia = clamp(ia, 2, p.n_a)
    ta = (a - p.a_grid[ia-1]) / (p.a_grid[ia] - p.a_grid[ia-1])
    lo = interp_ell(p.ell_grid, @view(W[ia-1, iz2, :]), ell)
    hi = interp_ell(p.ell_grid, @view(W[ia,   iz2, :]), ell)
    return (1.0 - ta) * lo + ta * hi
end


# =============================================================================
# SECTION 7: Firm Value Function (VFI on (z, ell) grid)
# =============================================================================

function solve_firm_vfi(theta::Float64, p::Params;
                        tol=1e-8, max_iter=2000, verbose=false)
    qt    = q_theta(theta, p)
    E     = zeros(p.n_z, p.n_ell)
    v_pol = zeros(p.n_z, p.n_ell)

    for iter in 1:max_iter
        E_new = similar(E)
        for iz in 1:p.n_z
            z = p.z_grid[iz]
            E_expect = zeros(p.n_ell)
            for iz2 in 1:p.n_z
                E_expect .+= p.Pi_z[iz, iz2] .* E[iz2, :]
            end
            for il in 1:p.n_ell
                ell      = p.ell_grid[il]
                w        = nash_wage(z, ell, theta, p)
                pi_flow  = pi_firm(z, ell, w, p)
                function firm_obj(v)
                    v        = max(v, 0.0)
                    ell_next = (1.0 - p.s) * ell + qt * v
                    cont     = interp_ell(p.ell_grid, E_expect, ell_next)
                    return -(pi_flow - p.kappa_v * v + p.beta * (1.0 - p.d) * cont)
                end
                v_max = (p.ell_grid[end] - (1.0 - p.s)*ell) / max(qt, 1e-8) + 1.0
                v_max = max(v_max, 0.0)
                if v_max < 1e-10
                    v_star = 0.0
                else
                    result = optimize(firm_obj, 0.0, v_max, Brent())
                    v_star = max(Optim.minimizer(result), 0.0)
                end
                ell_next      = (1.0 - p.s) * ell + qt * v_star
                cont          = interp_ell(p.ell_grid, E_expect, ell_next)
                E_new[iz, il] = pi_flow - p.kappa_v * v_star +
                                p.beta * (1.0 - p.d) * cont
                v_pol[iz, il] = v_star
            end
        end
        err = maximum(abs.(E_new .- E))
        E  .= E_new
        verbose && iter % 100 == 0 &&
            println(@sprintf("  firm VFI iter %4d | error = %.2e", iter, err))
        err < tol && break
    end
    return E, v_pol
end


# =============================================================================
# SECTION 8: Free Entry
# =============================================================================

function free_entry_residual(theta::Float64, p::Params; ell_entry::Float64=1.0)
    E, _ = solve_firm_vfi(theta, p)
    E_entry = sum(p.pi_z[iz] * interp_ell(p.ell_grid, E[iz, :], ell_entry)
                  for iz in 1:p.n_z)
    return E_entry - p.kappa_e
end

function bisect(f, lo::Float64, hi::Float64; tol=1e-6, max_iter=60, verbose=true)
    f_lo = f(lo); f_hi = f(hi)
    @assert f_lo * f_hi < 0 "bisect: f(lo) and f(hi) must have opposite signs"
    mid = (lo + hi) / 2.0
    for i in 1:max_iter
        mid   = (lo + hi) / 2.0
        f_mid = f(mid)
        verbose && println(@sprintf("  bisect iter %2d | theta = %.4f | residual = %+.4e", i, mid, f_mid))
        abs(f_mid) < tol && break
        f_lo * f_mid < 0 ? (hi = mid) : (lo = mid)
    end
    return mid
end


# =============================================================================
# SECTION 9: Worker Consumption-Savings Problem (BHA core)
#
# Each period a worker receives income y (= wage if employed, b if unemployed)
# and chooses next-period assets a' to solve:
#     c = (1+r)*a + y - a',   a' >= a_min,   c > 0
#
# Employed Bellman:
#   W(a,z_f,ell_f) = max_{a'} u(c) + beta*[ (1-delta)*E_z[W(a',z_f',ell_f')]
#                                          + delta * U(a') ]
# Unemployed Bellman:
#   U(a) = max_{a'} u(c) + beta*[ f(theta)*Wbar(a') + (1-f(theta))*U(a') ]
#
# delta    = s + d - s*d                          (job loss prob)
# ell_f'   = (1-s)*ell_f + q(theta)*v*(z_f,ell_f) (labor law of motion)
# Wbar(a') = vacancy-weighted average of W(a', z_f, ell_f) over firm states
#            (job finders are more likely to land at firms posting more vacancies)
#
# a' is chosen by CONTINUOUS optimization (Brent); the continuation value is
# interpolated over the asset (and ell) grids. This yields smooth policies.
# Policies are stored as the chosen a' VALUE (not a grid index).
# =============================================================================

function solve_worker_savings(theta::Float64, v_pol::Matrix{Float64}, p::Params;
                              tol=1e-7, max_iter=2000, verbose=false)
    qt    = q_theta(theta, p)
    delta = p.s + p.d - p.s * p.d
    ft    = f_theta(theta, p)

    W   = zeros(p.n_a, p.n_z, p.n_ell)
    U   = zeros(p.n_a)
    apW = zeros(p.n_a, p.n_z, p.n_ell)   # policy: chosen a' (value)
    apU = zeros(p.n_a)

    # Vacancy weights over firm states (where job finders land)
    Vw = [p.pi_z[iz] * v_pol[iz, il] for iz in 1:p.n_z, il in 1:p.n_ell]
    Vtot = sum(Vw)
    Vw = Vtot > 1e-10 ? Vw ./ Vtot : fill(1.0/(p.n_z*p.n_ell), p.n_z, p.n_ell)

    for iter in 1:max_iter
        W_new = similar(W)
        U_new = similar(U)

        # Wbar(a) on the asset grid: vacancy-weighted value a job seeker expects.
        # ACCEPTANCE / PARTICIPATION CONSTRAINT: a worker only accepts an offer
        # if employment beats staying unemployed, so the relevant value at each
        # firm state is max(W(a,z,ell), U(a)), not W itself. A worker offered a
        # job worse than unemployment rejects it and keeps U(a). By construction
        # this rules out accepted jobs with W < U.
        Wbar = zeros(p.n_a)
        for ia in 1:p.n_a
            acc = 0.0
            for iz in 1:p.n_z, il in 1:p.n_ell
                acc += Vw[iz, il] * max(W[ia, iz, il], U[ia])
            end
            Wbar[ia] = acc
        end

        # ---- Unemployed ----
        for ia in 1:p.n_a
            res   = (1.0 + p.r) * p.a_grid[ia] + p.b
            apmax = min(res - 1e-8, p.a_grid[end])
            if apmax <= p.a_min
                ap = p.a_min
                c  = res - ap
                U_new[ia] = util(c, p) + p.beta *
                    (ft * interp_a(Wbar, p, ap) + (1.0 - ft) * interp_a(U, p, ap))
                apU[ia] = ap
            else
                objU(ap) = -(util(res - ap, p) + p.beta *
                    (ft * interp_a(Wbar, p, ap) + (1.0 - ft) * interp_a(U, p, ap)))
                rr = optimize(objU, p.a_min, apmax, Brent())
                apU[ia]   = Optim.minimizer(rr)
                U_new[ia] = -Optim.minimum(rr)
            end
        end

        # ---- Employed ----
        for iz in 1:p.n_z
            for il in 1:p.n_ell
                ell      = p.ell_grid[il]
                w        = nash_wage(p.z_grid[iz], ell, theta, p)
                ell_next = (1.0 - p.s) * ell + qt * v_pol[iz, il]
                for ia in 1:p.n_a
                    res   = (1.0 + p.r) * p.a_grid[ia] + w
                    apmax = min(res - 1e-8, p.a_grid[end])

                    cont(ap) = begin
                        EW = 0.0
                        for iz2 in 1:p.n_z
                            EW += p.Pi_z[iz, iz2] * interp_W(W, p, ap, iz2, ell_next)
                        end
                        (1.0 - delta) * EW + delta * interp_a(U, p, ap)
                    end

                    if apmax <= p.a_min
                        ap = p.a_min
                        W_new[ia, iz, il] = util(res - ap, p) + p.beta * cont(ap)
                        apW[ia, iz, il]   = ap
                    else
                        objW(ap) = -(util(res - ap, p) + p.beta * cont(ap))
                        rr = optimize(objW, p.a_min, apmax, Brent())
                        apW[ia, iz, il]   = Optim.minimizer(rr)
                        W_new[ia, iz, il] = -Optim.minimum(rr)
                    end
                end
            end
        end

        err = max(maximum(abs.(W_new .- W)), maximum(abs.(U_new .- U)))
        W .= W_new
        U .= U_new
        verbose && iter % 25 == 0 &&
            println(@sprintf("  worker VFI iter %4d | error = %.2e", iter, err))
        err < tol && break
    end

    return W, U, apW, apU
end


# =============================================================================
# SECTION 10: Stationary Distribution & Aggregate Capital
#
# Simulate a panel of N workers for T periods, tracking assets as a CONTINUOUS
# variable and applying the interpolated savings policy each period.
# =============================================================================

function nearest_ell(p::Params, ell_val::Float64)
    ell_val = clamp(ell_val, p.ell_grid[1], p.ell_grid[end])
    i = searchsortedfirst(p.ell_grid, ell_val); i = clamp(i, 1, p.n_ell)
    (i > 1 && abs(p.ell_grid[i-1]-ell_val) < abs(p.ell_grid[i]-ell_val)) ? i-1 : i
end

function draw_next_z(p::Params, iz::Int)
    u = rand(); cum = 0.0
    for iz2 in 1:p.n_z
        cum += p.Pi_z[iz, iz2]
        u <= cum && return iz2
    end
    return p.n_z
end

function simulate_distribution(theta::Float64, v_pol::Matrix{Float64},
                               apW::Array{Float64,3}, apU::Vector{Float64},
                               W::Array{Float64,3}, U::Vector{Float64}, p::Params;
                               N=20_000, T=1_000, seed=1234)
    Random.seed!(seed)
    qt    = q_theta(theta, p)
    delta = p.s + p.d - p.s * p.d
    ft    = f_theta(theta, p)

    Vw = [p.pi_z[iz] * v_pol[iz, il] for iz in 1:p.n_z, il in 1:p.n_ell]
    Vtot = sum(Vw)
    Vw_flat = Vtot > 1e-10 ? vec(Vw) ./ Vtot : fill(1.0/(p.n_z*p.n_ell), p.n_z*p.n_ell)
    Vw_cum  = cumsum(Vw_flat)

    draw_firm() = begin
        u = rand()
        idx = clamp(searchsortedfirst(Vw_cum, u), 1, p.n_z*p.n_ell)
        iz  = ((idx - 1) % p.n_z) + 1
        il  = ((idx - 1) ÷ p.n_z) + 1
        (iz, il)
    end

    emp = falses(N)
    a   = fill(p.a_min, N)        # continuous assets
    fz  = ones(Int, N)
    fl  = ones(Int, N)

    for t in 1:T
        for n in 1:N
            if emp[n]
                # interpolated savings policy at current continuous assets
                a[n] = interp_a(@view(apW[:, fz[n], fl[n]]), p, a[n])
                ell_next = (1.0 - p.s) * p.ell_grid[fl[n]] + qt * v_pol[fz[n], fl[n]]
                fl[n] = nearest_ell(p, ell_next)
                fz[n] = draw_next_z(p, fz[n])
                rand() < delta && (emp[n] = false)
            else
                a[n] = interp_a(apU, p, a[n])
                if rand() < ft
                    iz, il = draw_firm()
                    # Acceptance constraint: take the job only if it beats
                    # staying unemployed at current assets (W >= U).
                    W_offer = interp_W(W, p, a[n], iz, p.ell_grid[il])
                    U_stay  = interp_a(U, p, a[n])
                    if W_offer >= U_stay
                        emp[n] = true
                        fz[n] = iz; fl[n] = il
                    end
                end
            end
        end
    end

    assets = copy(a)
    return (assets=assets, u_rate=count(!, emp)/N, K_agg=mean(assets),
            emp=emp, fz=fz, fl=fl)
end


# =============================================================================
# SECTION 11: Solve Full Equilibrium
# =============================================================================

struct Equilibrium
    theta :: Float64
    E     :: Matrix{Float64}
    v_pol :: Matrix{Float64}
    W     :: Array{Float64,3}
    U     :: Vector{Float64}
    apW   :: Array{Float64,3}
    apU   :: Vector{Float64}
    sim   :: NamedTuple
    p     :: Params
end

function solve_equilibrium(p::Params; theta_lo=0.05, theta_hi=8.0, ell_entry=1.0)
    println("=" ^ 64)
    println("Solving DMP + BHA equilibrium")
    println("=" ^ 64)

    println("\n[1/3] Free entry: solving for theta* ...")
    r_lo = free_entry_residual(theta_lo, p; ell_entry)
    r_hi = free_entry_residual(theta_hi, p; ell_entry)
    if r_lo * r_hi > 0
        error("No sign change in [theta_lo, theta_hi]; adjust bracket/parameters.")
    end
    theta_star = bisect(th -> free_entry_residual(th, p; ell_entry), theta_lo, theta_hi)
    E, v_pol   = solve_firm_vfi(theta_star, p)
    println(@sprintf("      theta* = %.4f", theta_star))

    println("\n[2/3] Worker consumption-savings (BHA) ...")
    W, U, apW, apU = solve_worker_savings(theta_star, v_pol, p; verbose=true)

    println("\n[3/3] Simulating stationary wealth distribution ...")
    sim = simulate_distribution(theta_star, v_pol, apW, apU, W, U, p)

    return Equilibrium(theta_star, E, v_pol, W, U, apW, apU, sim, p)
end


# =============================================================================
# SECTION 12: Report
# =============================================================================

# Representative employed worker's firm state: the MEDIAN (z, ell) among
# employed workers in the simulated stationary distribution. This avoids
# picking an arbitrary middle grid point that no worker may actually occupy.
function representative_firm(eq::Equilibrium)
    p = eq.p
    emp_idx = findall(eq.sim.emp)
    if isempty(emp_idx)
        return (p.n_z + 1) ÷ 2, (p.n_ell + 1) ÷ 2   # fallback
    end
    iz = clamp(round(Int, median(eq.sim.fz[emp_idx])), 1, p.n_z)
    il = clamp(round(Int, median(eq.sim.fl[emp_idx])), 1, p.n_ell)
    return iz, il
end

function report_equilibrium(eq::Equilibrium)
    p = eq.p; theta = eq.theta

    println("\n" * "=" ^ 64)
    println("EQUILIBRIUM RESULTS")
    println("=" ^ 64)

    println("\n--- Labor market ---")
    println(@sprintf("  theta*        = %.4f   [market tightness v/u]", theta))
    println(@sprintf("  f(theta*)     = %.4f   [job finding rate]",     f_theta(theta, p)))
    println(@sprintf("  q(theta*)     = %.4f   [vacancy filling rate]", q_theta(theta, p)))
    u_flow = p.s / (p.s + f_theta(theta, p))
    println(@sprintf("  u (flow)      = %.4f   [analytic unemployment rate]", u_flow))
    println(@sprintf("  u (simulated) = %.4f   [from panel]", eq.sim.u_rate))

    println("\n--- Wealth distribution (BHA) ---")
    a = sort(eq.sim.assets)
    qf(x) = a[clamp(round(Int, x*length(a)), 1, length(a))]
    println(@sprintf("  K (aggregate) = %.4f   [mean assets]", eq.sim.K_agg))
    println(@sprintf("  median assets = %.4f", qf(0.50)))
    println(@sprintf("  p10 / p90     = %.4f / %.4f", qf(0.10), qf(0.90)))
    frac = count(x -> x <= p.a_grid[1] + 1e-6, eq.sim.assets) / length(eq.sim.assets)
    println(@sprintf("  %% at borrowing constraint = %.1f%%", 100*frac))

    println("\n--- Savings policy spot-check (employed, representative firm) ---")
    iz, il = representative_firm(eq)
    w  = nash_wage(p.z_grid[iz], p.ell_grid[il], theta, p)
    println(@sprintf("  firm (z=%.3f, ell=%.2f), wage = %.4f", p.z_grid[iz], p.ell_grid[il], w))
    println(@sprintf("  %-10s %-10s %-10s", "a", "a'(saved)", "c"))
    println("  " * "-"^32)
    for ia in round.(Int, range(1, p.n_a, length=6))
        aa = p.a_grid[ia]; ap = eq.apW[ia, iz, il]
        c  = (1+p.r)*aa + w - ap
        println(@sprintf("  %-10.4f %-10.4f %-10.4f", aa, ap, c))
    end
end


# =============================================================================
# SECTION 12b: Visuals — save key plots as PNG files
#   Requires the Plots package:  ] add Plots
# =============================================================================

function make_plots(eq::Equilibrium; outdir::String=".")
    p = eq.p; theta = eq.theta

    # Precompute the wage at every firm state (for consumption)
    wage = [nash_wage(p.z_grid[iz], p.ell_grid[il], theta, p)
            for iz in 1:p.n_z, il in 1:p.n_ell]

    # 1. Wealth distribution
    plt1 = histogram(eq.sim.assets; bins=50, xlabel="assets a",
        ylabel="number of workers", title="Stationary Wealth Distribution", legend=false)
    savefig(plt1, joinpath(outdir, "wealth_distribution.png"))

    # 2. Savings policy a'(a): min and max across ALL firm (z,ell) states
    apW_min = [minimum(@view eq.apW[ia, :, :]) for ia in 1:p.n_a]
    apW_max = [maximum(@view eq.apW[ia, :, :]) for ia in 1:p.n_a]
    plt2 = plot(p.a_grid, apW_max; label="employed (best firm)", xlabel="assets today a",
        ylabel="assets tomorrow a'", title="Savings Policy", lw=2)
    plot!(plt2, p.a_grid, apW_min; label="employed (worst firm)", lw=2)
    plot!(plt2, p.a_grid, eq.apU;  label="unemployed", lw=2)
    plot!(plt2, p.a_grid, p.a_grid; label="45 degree", ls=:dash, lc=:gray)
    savefig(plt2, joinpath(outdir, "savings_policy.png"))

    # 3. Consumption policy c(a): min and max across ALL firm (z,ell) states
    cmat = [(1+p.r)*p.a_grid[ia] + wage[iz, il] - eq.apW[ia, iz, il]
            for ia in 1:p.n_a, iz in 1:p.n_z, il in 1:p.n_ell]
    c_min = [minimum(@view cmat[ia, :, :]) for ia in 1:p.n_a]
    c_max = [maximum(@view cmat[ia, :, :]) for ia in 1:p.n_a]
    c_un  = [(1+p.r)*p.a_grid[ia] + p.b - eq.apU[ia] for ia in 1:p.n_a]
    plt3 = plot(p.a_grid, c_max; label="employed (best firm)", xlabel="assets a",
        ylabel="consumption c", title="Consumption Policy", lw=2)
    plot!(plt3, p.a_grid, c_min; label="employed (worst firm)", lw=2)
    plot!(plt3, p.a_grid, c_un;  label="unemployed", lw=2)
    savefig(plt3, joinpath(outdir, "consumption_policy.png"))

    # 4. Value functions: min and max W across ALL firm (z,ell) states, vs U
    W_min = [minimum(@view eq.W[ia, :, :]) for ia in 1:p.n_a]
    W_max = [maximum(@view eq.W[ia, :, :]) for ia in 1:p.n_a]
    plt4 = plot(p.a_grid, W_max; label="W (best firm)", xlabel="assets a",
        ylabel="value", title="Value Functions (range across all firms)", lw=2)
    plot!(plt4, p.a_grid, W_min; label="W (worst firm)", lw=2)
    plot!(plt4, p.a_grid, eq.U;  label="U (unemployed)", lw=2, lc=:black, ls=:dash)
    savefig(plt4, joinpath(outdir, "value_functions.png"))

    # 5. Firm value and vacancies by productivity (at ell = 1)
    E_z = [interp_ell(p.ell_grid, eq.E[i, :], 1.0)     for i in 1:p.n_z]
    v_z = [interp_ell(p.ell_grid, eq.v_pol[i, :], 1.0) for i in 1:p.n_z]
    plt5 = plot(p.z_grid, E_z; label="firm value E(z, ell=1)", xlabel="productivity z",
        ylabel="value", title="Firm Value and Vacancies", lw=2)
    plot!(twinx(), p.z_grid, v_z; label="vacancies v(z)", lc=:red, lw=2, ylabel="vacancies")
    savefig(plt5, joinpath(outdir, "firm_value.png"))

    println("\nSaved 5 plots to: $(abspath(outdir))")
    return nothing
end


# =============================================================================
# SECTION 13: NEXT STEP — Occupational choice / entrepreneurship (NOT YET BUILT)
#
# To add the entrepreneurship layer:
#   - Give each agent their own ability z_self (a second Tauchen process)
#   - Add an entrepreneur value V_E(a, z_self) using their own capital a as k
#   - Unemployed Bellman becomes:  U(a, z_self) = max( search-as-worker,
#                                                      start-a-firm-as-entrepreneur )
#   - The borrowing constraint a' >= a_min then directly limits who can afford
#     to become an entrepreneur — this is the financial-development channel.
# =============================================================================

function main()
    p  = make_params()
    eq = solve_equilibrium(p)
    report_equilibrium(eq)
    make_plots(eq)
    return eq
end

eq = main();
nothing