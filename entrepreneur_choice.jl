# =============================================================================
# Occupational Choice & Entrepreneurship  (Stage 1)
# =============================================================================
#
# Builds the entrepreneurship layer on top of the DMP + BHA model. This is a
# STANDALONE stage-1 file: it keeps DMP_bha.jl intact and takes the labor-market
# objects (wage w, job-finding rate f) from the DMP equilibrium as inputs.
#
# Every agent now carries their own entrepreneurial ability z_self and assets a.
# There are three occupations:
#     unemployed (searching),  worker (employed),  entrepreneur (runs a firm)
#
# OCCUPATIONAL CHOICE (the heart of the model):
#   An unemployed agent compares the value of searching for a job (worker path)
#   with the value of starting a firm (entrepreneur path) and picks the larger:
#         V_unemp(a, z) = max( search-as-worker , start-a-firm )
#
# ENTREPRENEUR'S FIRM (Stage-1 modeling choice — FRICTIONLESS hiring):
#   An entrepreneur with ability z and assets a deploys capital k and hires
#   labor ell at the going wage w. To pin down firm size we use span-of-control
#   (decreasing returns), and capital is limited by a collateral constraint:
#         output  = z * ( k^alpha * ell^(1-alpha) )^nu        [nu < 1]
#         profit  = output - w*ell - r*k
#         k       <= lambda * a         [collateral / financial-access limit]
#   The collateral parameter lambda is the FINANCIAL DEVELOPMENT channel:
#   a higher lambda lets asset-poor but high-ability agents run bigger firms.
#
#   NOTE: this differs from the DMP firm block, where firms hire by posting
#   vacancies through the matching market. Whether entrepreneurs should hire
#   frictionlessly (here) or through the DMP market is a design choice to
#   confirm. Frictionless + span-of-control is the standard entrepreneurship
#   modeling approach (Buera-Shin style) and is the natural fit for this
#   research question.
#
#   On firm destruction (prob d_E) the entrepreneur loses a fraction (1-phi)
#   of deployed capital and returns to unemployment.
#
# NOT YET DONE (stage 2): folding this back into DMP_bha.jl so that the worker
# block also carries z_self and the whole thing solves as one three-loop
# equilibrium (inner: savings/labor; middle: occupation; outer: aggregates).
# =============================================================================

using LinearAlgebra, Statistics, Printf, Optim, Random, Plots

# =============================================================================
# SECTION 1: Parameters
# =============================================================================

struct EParams
    beta   :: Float64    # discount factor
    sigma  :: Float64    # CRRA risk aversion
    r      :: Float64    # return on assets / capital rental rate
    alpha  :: Float64    # capital share inside the production bundle
    nu     :: Float64    # span of control (decreasing returns), nu < 1
    lambda :: Float64    # collateral constraint: k <= lambda * a  (financial dev.)
    phi    :: Float64    # capital retained on firm destruction (scrap fraction)
    d_E    :: Float64    # entrepreneur firm-destruction probability
    cf     :: Float64    # fixed per-period operating cost of running a firm

    # labor-market objects taken from the DMP equilibrium:
    w      :: Float64    # going wage entrepreneurs pay / workers earn
    f_job  :: Float64    # job-finding rate for searching workers
    s      :: Float64    # job-separation rate for employed workers
    b      :: Float64    # unemployment benefit

    # ability process (Tauchen on log z)
    n_z    :: Int
    z_grid :: Vector{Float64}
    Pi_z   :: Matrix{Float64}

    # asset grid
    n_a    :: Int
    a_min  :: Float64
    a_max  :: Float64
    a_grid :: Vector{Float64}
end

function make_eparams(;
    beta   = 0.96,
    sigma  = 2.00,
    r      = 0.03,
    alpha  = 0.33,
    nu     = 0.85,      # < 1 so firm size is well-defined
    lambda = 1.50,      # collateral multiple of assets
    phi    = 0.90,      # keep 90% of capital on destruction
    d_E    = 0.05,      # entrepreneur destruction prob
    cf     = 0.50,      # fixed per-period operating cost (deters low-ability entry)
    w      = 0.80,      # <-- from DMP equilibrium (nash_wage at typical firm)
    f_job  = 0.90,      # <-- from DMP equilibrium (f_theta)
    s      = 0.03,
    b      = 0.40,
    n_z    = 7,
    rho_z  = 0.90,
    sig_z  = 0.20,
    n_a    = 80,
    a_min  = 0.0,       # borrowing constraint: a' >= 0 (no forced saving floor)
    a_max  = 50.0,
)
    z_grid, Pi_z, _ = tauchen_e(n_z, rho_z, sig_z)
    z_grid = exp.(z_grid)
    a_grid = a_min .+ (a_max - a_min) .* (range(0, 1, length=n_a)).^2
    return EParams(beta, sigma, r, alpha, nu, lambda, phi, d_E, cf,
                   w, f_job, s, b, n_z, z_grid, Pi_z, n_a, a_min, a_max, a_grid)
end

# Tauchen (same method as the main model)
function tauchen_e(n::Int, rho::Float64, sigma::Float64; m::Float64=3.0)
    su = sigma / sqrt(1 - rho^2)
    zg = collect(range(-m*su, m*su, length=n)); dz = zg[2]-zg[1]
    Pi = zeros(n, n)
    for i in 1:n, j in 1:n
        lo = zg[j]-dz/2; hi = zg[j]+dz/2; mu = rho*zg[i]
        Pi[i,j] = j==1 ? ncdf((hi-mu)/sigma) :
                  j==n ? 1-ncdf((lo-mu)/sigma) :
                         ncdf((hi-mu)/sigma)-ncdf((lo-mu)/sigma)
    end
    for i in 1:n; Pi[i,:] ./= sum(Pi[i,:]); end
    pi = fill(1.0/n, n)
    for _ in 1:2000
        pn = Pi'*pi; maximum(abs.(pn-pi))<1e-12 && break; pi = pn
    end
    return zg, Pi, pi
end

function erf_as(x::Float64)
    sg = sign(x); x = abs(x); t = 1/(1+0.3275911x)
    y = 1-(((((1.061405429t-1.453152027)t)+1.421413741)t-0.284496736)t+0.254829592)t*exp(-x*x)
    return sg*y
end
ncdf(x::Float64) = 0.5*(1+erf_as(x/sqrt(2)))

util(c::Float64, p::EParams) = c <= 1e-10 ? -1e6 :
    (abs(p.sigma-1) < 1e-8 ? log(c) : (c^(1-p.sigma)-1)/(1-p.sigma))


# =============================================================================
# SECTION 2: Entrepreneur's Static Problem
#
# Given ability z and assets a, choose capital k (<= lambda*a) and labor ell
# to maximize period profit:
#       profit = z*(k^alpha * ell^(1-alpha))^nu - w*ell - r*k
#
# Solved numerically: for each candidate k, the optimal ell has a closed form
# from the labor FOC; then optimize over k on [0, lambda*a].
# Returns (profit*, k*, ell*).
# =============================================================================

function entrepreneur_static(z::Float64, a::Float64, p::EParams)
    k_cap = p.lambda * a       # collateral limit on capital

    # optimal labor given k (labor FOC of the span-of-control problem)
    function best_ell(k)
        k <= 0 && return 0.0
        # profit_ell = z*(k^alpha)^nu * ell^((1-alpha)*nu) - w*ell - r*k
        A   = z * (k^p.alpha)^p.nu            # coefficient on ell^beta_l
        bl  = (1 - p.alpha) * p.nu            # labor exponent (< 1)
        # FOC: A*bl*ell^(bl-1) = w  ->  ell = (A*bl/w)^(1/(1-bl))
        return (A * bl / p.w)^(1.0 / (1.0 - bl))
    end

    function neg_profit(k)
        k = clamp(k, 0.0, k_cap)
        ell = best_ell(k)
        out = z * (k^p.alpha * ell^(1 - p.alpha))^p.nu
        return -(out - p.w * ell - p.r * k)
    end

    if k_cap < 1e-8
        return 0.0, 0.0, 0.0
    end
    res   = optimize(neg_profit, 0.0, k_cap, Brent())
    kstar = clamp(Optim.minimizer(res), 0.0, k_cap)
    estar = best_ell(kstar)
    pstar = z * (kstar^p.alpha * estar^(1 - p.alpha))^p.nu - p.w*estar - p.r*kstar
    return pstar, kstar, estar
end


# =============================================================================
# SECTION 3: Interpolation
# =============================================================================

function interp_a(V::AbstractVector, p::EParams, a::Float64)
    a = clamp(a, p.a_grid[1], p.a_grid[end])
    i = searchsortedfirst(p.a_grid, a); i = clamp(i, 2, p.n_a)
    t = (a - p.a_grid[i-1]) / (p.a_grid[i] - p.a_grid[i-1])
    return (1-t)*V[i-1] + t*V[i]
end


# =============================================================================
# SECTION 4: Value Functions with Occupational Choice
#
# Three value functions, each over (a, z):
#   Vu(a,z) = unemployed:  max_{a'} u(c) + beta*E_z[ Vocc(a', z') ]
#             where this period a searching agent gets benefit b and either
#             finds a job (prob f_job -> worker next period if it beats searching)
#             or keeps searching. The OCCUPATION decision is embedded in Vocc.
#   Vw(a,z) = worker:      earns w, separates to unemployment at rate s
#   Ve(a,z) = entrepreneur: earns profit*(z,a), firm destroyed at rate d_E
#             (loses (1-phi) of capital, returns to unemployment)
#
# Occupational choice each period for a NON-employed agent:
#   Vocc(a,z) = max( Vu_search(a,z), Ve(a,z) )
# We implement this by letting the unemployed compare search vs entrepreneurship.
# =============================================================================

function solve_occupation(p::EParams; tol=1e-7, max_iter=2000, verbose=false)
    nz, na = p.n_z, p.n_a

    Vu = zeros(na, nz)   # value of being unemployed and CHOOSING occupation
    Vw = zeros(na, nz)   # value of being an employed worker
    Ve = zeros(na, nz)   # value of being an entrepreneur

    apU = zeros(na, nz)  # savings policies
    apW = zeros(na, nz)
    apE = zeros(na, nz)
    occ = zeros(Int, na, nz)   # 1 = search/worker, 2 = entrepreneur (chosen when unemployed)

    # precompute entrepreneur static profit, capital, labor on the grid
    prof = zeros(na, nz); kpol = zeros(na, nz); lpol = zeros(na, nz)
    for iz in 1:nz, ia in 1:na
        prof[ia,iz], kpol[ia,iz], lpol[ia,iz] =
            entrepreneur_static(p.z_grid[iz], p.a_grid[ia], p)
    end

    for iter in 1:max_iter
        Vu_new = similar(Vu); Vw_new = similar(Vw); Ve_new = similar(Ve)

        for iz in 1:nz
            # expected next-period value over z', for each occupation
            EVu = zeros(na); EVw = zeros(na); EVe = zeros(na)
            for iz2 in 1:nz
                EVu .+= p.Pi_z[iz,iz2] .* Vu[:,iz2]
                EVw .+= p.Pi_z[iz,iz2] .* Vw[:,iz2]
                EVe .+= p.Pi_z[iz,iz2] .* Ve[:,iz2]
            end

            for ia in 1:na
                a = p.a_grid[ia]

                # ---------- WORKER ----------
                resW = (1+p.r)*a + p.w
                apmaxW = min(resW - 1e-8, p.a_grid[end])
                contW(ap) = (1-p.s)*interp_a(EVw,p,ap) + p.s*interp_a(EVu,p,ap)
                if apmaxW <= p.a_min
                    ap = p.a_min
                    Vw_new[ia,iz] = util(resW-ap,p) + p.beta*contW(ap); apW[ia,iz]=ap
                else
                    rr = optimize(ap -> -(util(resW-ap,p)+p.beta*contW(ap)), p.a_min, apmaxW, Brent())
                    apW[ia,iz] = Optim.minimizer(rr); Vw_new[ia,iz] = -Optim.minimum(rr)
                end

                # ---------- ENTREPRENEUR ----------
                # income = profit minus fixed operating cost cf; on destruction
                # lose (1-phi)*k and go unemployed
                resE = (1+p.r)*a + prof[ia,iz] - p.cf
                apmaxE = min(resE - 1e-8, p.a_grid[end])
                # destruction: assets next period reduced by (1-phi)*k, then unemployed
                a_after_destruction(ap) = max(ap - (1-p.phi)*kpol[ia,iz], p.a_min)
                contE(ap) = (1-p.d_E)*interp_a(EVe,p,ap) +
                            p.d_E*interp_a(EVu,p,a_after_destruction(ap))
                if apmaxE <= p.a_min
                    ap = p.a_min
                    Ve_new[ia,iz] = util(resE-ap,p) + p.beta*contE(ap); apE[ia,iz]=ap
                else
                    rr = optimize(ap -> -(util(resE-ap,p)+p.beta*contE(ap)), p.a_min, apmaxE, Brent())
                    apE[ia,iz] = Optim.minimizer(rr); Ve_new[ia,iz] = -Optim.minimum(rr)
                end

                # ---------- UNEMPLOYED (searches) + OCCUPATIONAL CHOICE ----------
                # This period: benefit b, choose savings. Next period: find a job
                # (prob f_job) and become a worker if that beats searching; else
                # remain unemployed. The entrepreneurship option is taken now if
                # Ve exceeds the search value (occupational choice).
                resU = (1+p.r)*a + p.b
                apmaxU = min(resU - 1e-8, p.a_grid[end])
                contU(ap) = p.f_job*max(interp_a(EVw,p,ap), interp_a(EVu,p,ap)) +
                            (1-p.f_job)*interp_a(EVu,p,ap)
                if apmaxU <= p.a_min
                    ap = p.a_min
                    search_val = util(resU-ap,p) + p.beta*contU(ap); apU[ia,iz]=ap
                else
                    rr = optimize(ap -> -(util(resU-ap,p)+p.beta*contU(ap)), p.a_min, apmaxU, Brent())
                    apU[ia,iz] = Optim.minimizer(rr); search_val = -Optim.minimum(rr)
                end

                # OCCUPATIONAL CHOICE: search vs become entrepreneur
                if Ve_new[ia,iz] > search_val
                    Vu_new[ia,iz] = Ve_new[ia,iz]
                    occ[ia,iz]    = 2          # chooses entrepreneurship
                else
                    Vu_new[ia,iz] = search_val
                    occ[ia,iz]    = 1          # chooses to search for work
                end
            end
        end

        err = max(maximum(abs.(Vu_new-Vu)),
                  maximum(abs.(Vw_new-Vw)),
                  maximum(abs.(Ve_new-Ve)))
        Vu .= Vu_new; Vw .= Vw_new; Ve .= Ve_new
        verbose && iter % 25 == 0 &&
            println(@sprintf("  occ VFI iter %4d | error = %.2e", iter, err))
        err < tol && break
    end

    return (Vu=Vu, Vw=Vw, Ve=Ve, occ=occ,
            prof=prof, kpol=kpol, lpol=lpol,
            apU=apU, apW=apW, apE=apE, p=p)
end


# =============================================================================
# SECTION 5: Report & Plots
# =============================================================================

function report_occupation(sol)
    p = sol.p
    println("\n" * "="^60)
    println("OCCUPATIONAL CHOICE — RESULTS")
    println("="^60)

    # fraction of (a,z) grid where entrepreneurship is chosen
    frac_ent = count(==(2), sol.occ) / length(sol.occ)
    println(@sprintf("\n  Entrepreneurship chosen on %.1f%% of the (a,z) grid", 100*frac_ent))

    println("\n  Entrepreneurship cutoff: minimum assets to start a firm, by ability z")
    println(@sprintf("  %-12s %-16s", "z", "min assets a*"))
    println("  " * "-"^28)
    for iz in 1:p.n_z
        ia = findfirst(==(2), sol.occ[:, iz])
        cutoff = ia === nothing ? NaN : p.a_grid[ia]
        println(@sprintf("  %-12.4f %-16s", p.z_grid[iz],
                isnan(cutoff) ? "never" : @sprintf("%.4f", cutoff)))
    end
end

function plot_occupation(sol; outdir::String=".")
    p = sol.p

    # 1. Occupational choice map in (a, z) space
    #    heatmap: 1 = search/worker, 2 = entrepreneur
    plt1 = heatmap(p.a_grid, p.z_grid, sol.occ';
        xlabel="assets a", ylabel="ability z",
        title="Occupational Choice (yellow = entrepreneur)",
        colorbar=false)
    savefig(plt1, joinpath(outdir, "occupation_map.png"))

    # 2. Entrepreneur firm size (labor hired) by ability, at median assets
    ia_med = (p.n_a + 1) ÷ 2
    ell_by_z = [sol.lpol[ia_med, iz] for iz in 1:p.n_z]
    plt2 = plot(p.z_grid, ell_by_z; lw=2, legend=false,
        xlabel="ability z", ylabel="workers hired ell",
        title="Entrepreneur Firm Size by Ability (median assets)")
    savefig(plt2, joinpath(outdir, "entrepreneur_size.png"))

    # 3. Value functions vs assets at median ability
    #    (start at index 2: a=0 is degenerate for an entrepreneur — zero wealth
    #     means zero capital and zero output, so its value is not meaningful)
    iz_med = (p.n_z + 1) ÷ 2
    rng = 2:p.n_a
    plt3 = plot(p.a_grid[rng], sol.Vw[rng, iz_med]; label="worker", lw=2,
        xlabel="assets a", ylabel="value",
        title="Value by Occupation (median ability z)")
    plot!(plt3, p.a_grid[rng], sol.Ve[rng, iz_med]; label="entrepreneur", lw=2)
    plot!(plt3, p.a_grid[rng], sol.Vu[rng, iz_med]; label="unemployed (max)", lw=2, ls=:dash, lc=:black)
    savefig(plt3, joinpath(outdir, "occupation_values.png"))

    println("\nSaved 3 plots to: $(abspath(outdir))")
    return nothing
end


# =============================================================================
# SECTION 6: Main
# =============================================================================

function main_occupation()
    p = make_eparams()
    println("Solving occupational-choice value functions ...")
    sol = solve_occupation(p; verbose=true)
    report_occupation(sol)
    plot_occupation(sol)
    return sol
end

sol = main_occupation();
nothing