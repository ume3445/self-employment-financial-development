# =============================================================================
# DMP Model with Multi-Worker Firms
# =============================================================================
#
# Agents:   Unemployed workers, Employed workers, Firms
# State:    Firm state (z, ell) — productivity and size
# Choice:   Firms post vacancies v; workers accept/reject via Nash bargaining
#
# Value functions:
#   U        = value of being unemployed
#   W(z,ell) = value of being employed at a firm (z, ell)
#   E(z,ell) = value of a firm with productivity z and ell workers
#
# Production (Cobb-Douglas):
#   f(z, ell, k) = z * k^alpha * ell^(1-alpha)
#
# Matching function (Cobb-Douglas):
#   mu(u, v) = A * u^xi * v^(1-xi)
#   f_theta(theta) = mu/u = A * theta^(1-xi)     [job finding rate]
#   q_theta(theta) = mu/v = A * theta^(-xi)       [vacancy filling rate]
#   theta           = v/u                         [market tightness]
#
# Wages set by Nash bargaining with worker power eta.
# Equilibrium theta pinned down by free entry: E(z_bar, 0) = kappa_e
#
# =============================================================================

using LinearAlgebra, Statistics, Printf, Optim

# =============================================================================
# SECTION 1: Parameters
# =============================================================================

struct Params
    # --- Preferences ---
    beta    :: Float64      # discount factor
    b       :: Float64      # unemployment benefit

    # --- Production ---
    alpha   :: Float64      # capital share in f(z,ell,k) = z * k^alpha * ell^(1-alpha)
    k       :: Float64      # capital per firm (fixed/normalized)
    r       :: Float64      # rental rate of capital

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
    Pi_z     :: Matrix{Float64}  # n_z x n_z transition matrix
    pi_z     :: Vector{Float64}  # stationary distribution of z

    # --- Firm size grid ---
    n_ell    :: Int
    ell_grid :: Vector{Float64}
end

function make_params(;
    beta    = 0.99,
    b       = 0.40,
    alpha   = 0.33,
    k       = 1.00,
    r       = 0.01,
    s       = 0.03,
    d       = 0.02,
    eta     = 0.50,
    kappa_v = 0.50,
    kappa_e = 1.00,
    A       = 0.70,
    xi      = 0.50,
    n_z     = 10,
    rho_z   = 0.90,
    sigma_z = 0.10,
    n_ell   = 40,
    ell_max = 15.0,
)
    z_grid, Pi_z, pi_z = tauchen(n_z, rho_z, sigma_z)
    z_grid = exp.(z_grid)       # exponentiate: z is log-normally distributed

    ell_grid = collect(range(0.1, ell_max, length=n_ell))

    return Params(beta, b, alpha, k, r, s, d, eta, kappa_v, kappa_e,
                  A, xi, n_z, z_grid, Pi_z, pi_z, n_ell, ell_grid)
end


# =============================================================================
# SECTION 2: Tauchen (1986) Discretization of AR(1) Process
#
# log(z') = rho_z * log(z) + eps,   eps ~ N(0, sigma_z^2)
# Returns: z_grid (n_z,), Pi_z (n_z x n_z), pi_z (n_z,)
# =============================================================================

function tauchen(n::Int, rho::Float64, sigma::Float64; m::Float64=3.0)
    sigma_unc = sigma / sqrt(1.0 - rho^2)      # unconditional std of log z

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
        Pi[i, :] ./= sum(Pi[i, :])     # normalize rows to sum to 1
    end

    # Stationary distribution: iterate Pi' until convergence
    pi = fill(1.0/n, n)
    for _ in 1:2000
        pi_new = Pi' * pi
        maximum(abs.(pi_new - pi)) < 1e-12 && break
        pi = pi_new
    end

    return z_grid, Pi, pi
end

# Standard normal CDF (no external packages needed)
function normal_cdf(x::Float64)
    return 0.5 * erfc(-x / sqrt(2.0))
end


# =============================================================================
# SECTION 3: Matching Functions and Market Tightness
#
# mu(u, v) = A * u^xi * v^(1-xi)
# f_theta(theta) = A * theta^(1-xi)    [per-period job finding rate]
# q_theta(theta) = A * theta^(-xi)     [per-period vacancy filling rate]
# =============================================================================

f_theta(theta::Float64, p::Params) = p.A * theta^(1.0 - p.xi)
q_theta(theta::Float64, p::Params) = p.A * theta^(-p.xi)


# =============================================================================
# SECTION 4: Production, MPL, and Profits
#
# Production function (Cobb-Douglas):
#   f(z, ell, k) = z * k^alpha * ell^(1-alpha)
#
# Marginal product of labor:
#   MPL(z, ell) = (1-alpha) * z * k^alpha * ell^(-alpha)
#
# Profits (before vacancy posting costs):
#   pi(z, ell, w) = f(z, ell, k) - w*ell - r*k
# =============================================================================

f_prod(z::Float64, ell::Float64, p::Params) =
    z * p.k^p.alpha * ell^(1.0 - p.alpha)

MPL(z::Float64, ell::Float64, p::Params) =
    (1.0 - p.alpha) * z * p.k^p.alpha * ell^(-p.alpha)

pi_firm(z::Float64, ell::Float64, w::Float64, p::Params) =
    f_prod(z, ell, p) - w * ell - p.r * p.k


# =============================================================================
# SECTION 5: Nash Bargaining Wage
#
# w = eta * (MPL + kappa_v * theta) + (1 - eta) * b
#
# Worker gets fraction eta of their marginal contribution (MPL plus the
# savings from not having to fill another vacancy), and fraction (1-eta)
# of their outside option (unemployment benefit b).
# =============================================================================

function nash_wage(z::Float64, ell::Float64, theta::Float64, p::Params)
    mpl = MPL(z, ell, p)
    return p.eta * (mpl + p.kappa_v * theta) + (1.0 - p.eta) * p.b
end


# =============================================================================
# SECTION 6: Worker Value Functions
#
# W(z, ell): value of being employed at a firm with productivity z, size ell
#
# The correct Bellman equation — derived from professor's notes — is:
#
#   W(z, ell) = w(z,ell) + beta * [ delta*U  +  (1-delta) * E_z[W(z', ell')] ]
#
# where:
#   delta    = s + d - s*d          [job loss prob: separation OR firm destruction]
#   ell'     = (1-s)*ell + q(theta)*v*(z,ell)   [law of motion for labor]
#   E_z[W(z', ell')] = sum_{z'} Pi_z[z, z'] * W(z', ell')   [expectation over z']
#
# Two things evolve next period:
#   (1) z evolves according to the Markov process Pi_z
#   (2) ell evolves using the firm's vacancy posting policy v*(z,ell)
#
# This is solved by VFI on the full (n_z x n_ell) grid,
# given v_pol from the firm problem and a value for U.
#
# U: value of being unemployed
#   U = b + beta * [f_theta(theta)*W_bar + (1-f_theta(theta))*U]
#   => U = (b + beta*f_theta*W_bar) / (1 - beta*(1-f_theta))
#
# W_bar: vacancy-weighted average worker value
#   When a worker finds a job, they are more likely to match with a firm
#   posting MORE vacancies. So W_bar weights W(z,ell) by v*(z,ell)*pi_z(z).
#   W_bar = sum_{z,ell} [pi_z(z)*v*(z,ell) / V_total] * W(z,ell)
#   where V_total = sum_{z,ell} pi_z(z)*v*(z,ell)
#
# W and U are solved jointly via an outer fixed-point loop on U.
# =============================================================================

function solve_worker_vfi(theta::Float64, v_pol::Matrix{Float64},
                          U::Float64, p::Params;
                          tol=1e-8, max_iter=2000)

    qt      = q_theta(theta, p)
    W       = zeros(p.n_z, p.n_ell)
    delta   = p.s + p.d - p.s * p.d    # job loss probability (sep OR destruction)

    for iter in 1:max_iter
        W_new = similar(W)

        for iz in 1:p.n_z
            # E_z[W(z', ell) | z] for each ell point on the grid
            # = sum_{z'} Pi_z[iz, iz'] * W[iz', :]
            W_expect = zeros(p.n_ell)
            for iz2 in 1:p.n_z
                W_expect .+= p.Pi_z[iz, iz2] .* W[iz2, :]
            end

            for il in 1:p.n_ell
                ell = p.ell_grid[il]
                w   = nash_wage(p.z_grid[iz], ell, theta, p)

                # Law of motion for ell using firm's optimal vacancy policy
                ell_next = (1.0 - p.s) * ell + qt * v_pol[iz, il]

                # E_z[W(z', ell') | z] evaluated at ell'
                W_next = interp_ell(p.ell_grid, W_expect, ell_next)

                # Bellman: W(z,ell) = w + beta*[delta*U + (1-delta)*E_z[W(z',ell')]]
                W_new[iz, il] = w + p.beta * (delta * U + (1.0 - delta) * W_next)
            end
        end

        err = maximum(abs.(W_new .- W))
        W  .= W_new
        err < tol && break
    end

    return W
end

function unemployed_value(theta::Float64, W_bar::Float64, p::Params)
    ft = f_theta(theta, p)
    return (p.b + p.beta * ft * W_bar) / (1.0 - p.beta * (1.0 - ft))
end

# Solve W and U jointly given theta and v_pol
function solve_worker_values(theta::Float64, v_pol::Matrix{Float64}, p::Params;
                             ell_entry::Float64=1.0, tol=1e-8, max_iter=500)

    U = p.b / (1.0 - p.beta)       # initial guess: PV of unemployment benefit forever
    W = zeros(p.n_z, p.n_ell)

    for _ in 1:max_iter

        # Step 1: solve W(z,ell) via VFI given current U
        W = solve_worker_vfi(theta, v_pol, U, p)

        # Step 2: vacancy-weighted W_bar
        # f needs to be a vector (professor's note): weight each (z,ell)
        # by pi_z(z) * v*(z,ell) so firms posting more vacancies get more weight
        V_weights = [p.pi_z[iz] * v_pol[iz, il]
                     for iz in 1:p.n_z, il in 1:p.n_ell]
        V_total   = sum(V_weights)

        if V_total > 1e-10
            W_bar = sum(V_weights[iz, il] * W[iz, il]
                        for iz in 1:p.n_z, il in 1:p.n_ell) / V_total
        else
            # fallback if no vacancies posted
            W_bar = sum(p.pi_z[iz] * interp_ell(p.ell_grid, W[iz, :], ell_entry)
                        for iz in 1:p.n_z)
        end

        # Step 3: update U
        U_new = unemployed_value(theta, W_bar, p)

        abs(U_new - U) < tol && return U_new, W_bar, W
        U = 0.5 * U + 0.5 * U_new      # dampened update for stability
    end

    # Final W_bar after max iterations
    V_weights = [p.pi_z[iz] * v_pol[iz, il]
                 for iz in 1:p.n_z, il in 1:p.n_ell]
    V_total   = sum(V_weights)
    W_bar = V_total > 1e-10 ?
        sum(V_weights[iz, il] * W[iz, il]
            for iz in 1:p.n_z, il in 1:p.n_ell) / V_total :
        sum(p.pi_z[iz] * interp_ell(p.ell_grid, W[iz, :], ell_entry)
            for iz in 1:p.n_z)

    return U, W_bar, W
end


# =============================================================================
# SECTION 7: Firm Value Function (VFI on (z, ell) grid)
#
# E(z, ell): value of a firm with productivity z and ell workers
#
# Each period:
#   1. Produce f(z, ell, k), pay wages w*ell, pay capital cost r*k
#   2. Choose vacancies v >= 0 at cost kappa_v * v
#   3. Next period workforce: ell' = (1-s)*ell + q_theta(theta)*v
#   4. Firm survives with prob (1-d); destroyed with prob d (gets 0)
#   5. z evolves according to Pi_z (Markov process)
#
# Bellman equation:
#   E(z, ell) = max_{v>=0} [ pi(z,ell,w) - kappa_v*v
#                           + beta*(1-d) * E_z[E(z', ell') | z] ]
#
# Solved by VFI over the (n_z x n_ell) grid.
# Linear interpolation handles off-grid ell' values.
# =============================================================================

function interp_ell(ell_grid::Vector{Float64}, V::Vector{Float64}, ell::Float64)
    ell = clamp(ell, ell_grid[1], ell_grid[end])
    i   = searchsortedfirst(ell_grid, ell)
    i   = clamp(i, 2, length(ell_grid))
    t   = (ell - ell_grid[i-1]) / (ell_grid[i] - ell_grid[i-1])
    return (1.0 - t) * V[i-1] + t * V[i]
end

function solve_firm_vfi(theta::Float64, p::Params;
                        tol=1e-8, max_iter=2000, verbose=false)

    qt    = q_theta(theta, p)
    E     = zeros(p.n_z, p.n_ell)
    v_pol = zeros(p.n_z, p.n_ell)

    for iter in 1:max_iter
        E_new = similar(E)

        for iz in 1:p.n_z
            z = p.z_grid[iz]

            # E_z[E(z', ell) | z] for each ell: expectation over z'
            E_expect = zeros(p.n_ell)
            for iz2 in 1:p.n_z
                E_expect .+= p.Pi_z[iz, iz2] .* E[iz2, :]
            end

            for il in 1:p.n_ell
                ell      = p.ell_grid[il]
                w        = nash_wage(z, ell, theta, p)
                pi_flow  = pi_firm(z, ell, w, p)   # f(z,ell,k) - w*ell - r*k

                # Firm chooses v to maximize:
                #   pi_flow - kappa_v*v + beta*(1-d)*E_z[E(z', ell') | z]
                # subject to ell' = (1-s)*ell + q_theta*v
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
            println(@sprintf("  VFI iter %4d | error = %.2e", iter, err))
        err < tol && break
    end

    return E, v_pol
end


# =============================================================================
# SECTION 8: Free Entry Condition
#
# Firms enter by paying kappa_e. In equilibrium:
#   E_entry(theta) = sum_z pi_z(z) * E(z, ell_entry) = kappa_e
#
# This pins down equilibrium theta*. Solved via bisection.
# =============================================================================

function free_entry_residual(theta::Float64, p::Params; ell_entry::Float64=1.0)
    E, _ = solve_firm_vfi(theta, p)
    E_entry = sum(p.pi_z[iz] * interp_ell(p.ell_grid, E[iz, :], ell_entry)
                  for iz in 1:p.n_z)
    return E_entry - p.kappa_e
end

function bisect(f, lo::Float64, hi::Float64; tol=1e-6, max_iter=60)
    f_lo = f(lo)
    f_hi = f(hi)
    @assert f_lo * f_hi < 0 "bisect: f(lo) and f(hi) must have opposite signs"

    mid = (lo + hi) / 2.0
    for i in 1:max_iter
        mid   = (lo + hi) / 2.0
        f_mid = f(mid)
        println(@sprintf("  bisect iter %2d | theta = %.4f | residual = %+.4e", i, mid, f_mid))
        abs(f_mid) < tol && break
        f_lo * f_mid < 0 ? (hi = mid) : (lo = mid)
    end
    return mid
end


# =============================================================================
# SECTION 9: Solve Equilibrium
# =============================================================================

struct Equilibrium
    theta   :: Float64
    E       :: Matrix{Float64}
    v_pol   :: Matrix{Float64}
    W       :: Matrix{Float64}
    U       :: Float64
    W_bar   :: Float64
    p       :: Params
end

function solve_equilibrium(p::Params;
                           theta_lo=0.05, theta_hi=8.0, ell_entry=1.0)

    println("=" ^ 60)
    println("Solving DMP equilibrium via free entry condition...")
    println("=" ^ 60)

    r_lo = free_entry_residual(theta_lo, p; ell_entry)
    r_hi = free_entry_residual(theta_hi, p; ell_entry)
    println(@sprintf("Residual at theta=%.2f: %+.4f", theta_lo, r_lo))
    println(@sprintf("Residual at theta=%.2f: %+.4f", theta_hi, r_hi))

    if r_lo * r_hi > 0
        error("No sign change in [theta_lo, theta_hi]. Adjust bracket or parameters.")
    end

    theta_star = bisect(
        th -> free_entry_residual(th, p; ell_entry),
        theta_lo, theta_hi
    )

    println("\nRecomputing at theta* = $(round(theta_star, digits=4))")
    E, v_pol        = solve_firm_vfi(theta_star, p; verbose=true)
    U, W_bar, W     = solve_worker_values(theta_star, v_pol, p; ell_entry)

    return Equilibrium(theta_star, E, v_pol, W, U, W_bar, p)
end


# =============================================================================
# SECTION 10: Report Equilibrium
# =============================================================================

function report_equilibrium(eq::Equilibrium)
    p     = eq.p
    theta = eq.theta

    println("\n" * "=" ^ 60)
    println("EQUILIBRIUM RESULTS")
    println("=" ^ 60)

    println("\n--- Market Tightness ---")
    println(@sprintf("  theta*        = %.4f   [market tightness v/u]",   theta))
    println(@sprintf("  f_theta*      = %.4f   [job finding rate]",        f_theta(theta, p)))
    println(@sprintf("  q_theta*      = %.4f   [vacancy filling rate]",    q_theta(theta, p)))
    u_rate = p.s / (p.s + f_theta(theta, p))
    println(@sprintf("  u*            = %.4f   [unemployment rate]",       u_rate))

    println("\n--- Worker Values ---")
    println(@sprintf("  U             = %.4f   [unemployed value]",        eq.U))
    println(@sprintf("  W_bar         = %.4f   [vacancy-weighted avg W]",  eq.W_bar))
    println(@sprintf("  W_bar - U     = %.4f   [worker surplus]",          eq.W_bar - eq.U))

    println("\n--- Wages, MPL, and Profits at ell=1, by z ---")
    println(@sprintf("  %-12s %-10s %-10s %-10s", "z", "wage", "MPL", "pi(z,1)"))
    println("  " * "-" ^ 44)
    for iz in round.(Int, range(1, p.n_z, length=5))
        z  = p.z_grid[iz]
        w  = nash_wage(z, 1.0, theta, p)
        println(@sprintf("  %-12.4f %-10.4f %-10.4f %-10.4f",
                z, w, MPL(z, 1.0, p), pi_firm(z, 1.0, w, p)))
    end

    println("\n--- Firm Value E(z, ell=1) ---")
    println(@sprintf("  %-12s %-10s %-10s", "z", "E(z,1)", "v*(z,1)"))
    println("  " * "-" ^ 34)
    for iz in round.(Int, range(1, p.n_z, length=5))
        z = p.z_grid[iz]
        println(@sprintf("  %-12.4f %-10.4f %-10.4f",
                z,
                interp_ell(p.ell_grid, eq.E[iz, :], 1.0),
                interp_ell(p.ell_grid, eq.v_pol[iz, :], 1.0)))
    end

    println("\n--- Free Entry Check ---")
    E_entry = sum(p.pi_z[iz] * interp_ell(p.ell_grid, eq.E[iz, :], 1.0)
                  for iz in 1:p.n_z)
    println(@sprintf("  E_entry(theta*) = %.4f  (should equal kappa_e = %.4f)",
                     E_entry, p.kappa_e))
    println(@sprintf("  Residual        = %.2e", E_entry - p.kappa_e))
end


# =============================================================================
# SECTION 11: Main
# =============================================================================

function main()
    p = make_params(
        beta    = 0.99,
        b       = 0.40,
        alpha   = 0.33,
        k       = 1.00,
        r       = 0.01,
        s       = 0.03,
        d       = 0.02,
        eta     = 0.50,
        kappa_v = 0.50,
        kappa_e = 1.00,
        A       = 0.70,
        xi      = 0.50,
        n_z     = 10,
        rho_z   = 0.90,
        sigma_z = 0.10,
        n_ell   = 40,
        ell_max = 15.0,
    )

    eq = solve_equilibrium(p; theta_lo=0.05, theta_hi=8.0)
    report_equilibrium(eq)
    return eq
end

eq = main()