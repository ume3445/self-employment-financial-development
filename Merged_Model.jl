# =============================================================================
# Merged Model: DMP + BHA + Occupational Choice (Stage A)
# =============================================================================
#
# One model that combines the three pieces built separately:
#   - DMP firms with multi-worker hiring through a matching market
#   - Bewley-Huggett-Aiyagari savings with a borrowing constraint
#   - Occupational choice: be a worker or an entrepreneur
#
# A PERSON is defined by (assets a, ability z). Each period:
#   - Unemployed: choose to SEARCH for a wage job, or START a firm (entrepreneur).
#   - Worker: keep working, or QUIT to start a firm.
#   - Entrepreneur: run the firm; it grows by posting vacancies and hiring
#     through the DMP matching market (Nash-bargained wages). A new firm
#     starts with ZERO workers and grows over time. On destruction the
#     entrepreneur returns to unemployment.
#
# DESIGN DECISIONS (confirmed):
#   1a  Worker state = (own assets a, own ability z, wage earned w)
#   2a  A person's ability z IS the productivity of the firm they would run
#   3b  Entrepreneurs hire through the DMP matching market (like firms)
#   4   Wages are Nash-bargained
#   5a  Solve the firm/labor-market side first (theta, wage), then households
#   6b  Notation: Vu (unemployed), Vw (worker), Ve (entrepreneur)
#   7   Output: value functions, policies, occupation map + plots
#       (population simulation is the next stage, not here)
#
# SOLVE ORDER (5a, justified by the strike threat point):
#   Because wages bargain against b (strike), not U, the firm's hiring problem
#   and market tightness theta do NOT depend on the wealth distribution. So we
#   can solve the labor-market block first, then take w(z,ell) and theta as
#   given when solving the household occupational-choice block.
#
# CAPITAL & THE COLLATERAL CONSTRAINT (Change 1, the financial-development core):
#   Capital is CHOSEN, not fixed. An entrepreneur of productivity z running a
#   firm of size ell would optimally deploy k*(z,ell) solving MPK = u_k, where
#   the user cost u_k = r + delta + d (opportunity cost + depreciation + the
#   capital lost when the firm is destroyed). But capital is limited by the
#   entrepreneur's own assets via a collateral constraint:
#        k_eff = min(k*(z,ell), lambda * a).
#   So a talented-but-poor entrepreneur cannot deploy the capital they'd want.
#   Production has decreasing returns (span of control nu<1):
#        output = z * (k^alpha * ell^(1-alpha))^nu .
#   nu<1 is REQUIRED once k is a choice: under constant returns the optimal k
#   scales linearly with ell, output becomes linear in ell, and firm size is
#   indeterminate. The old fixed-k model is the nu=1 special case.
#
# SOLVE ORDER UNDER THE CONSTRAINT (honest caveat to 5a):
#   k_eff depends on the OWNER's assets a, so strictly the firm's profit, wage,
#   and hiring now depend on a and 5a no longer holds exactly. We PRESERVE the
#   solve order as a partial-equilibrium approximation: the labor block (theta,
#   wage schedule w(z,ell), firm value, hiring policy v_pol) is solved at the
#   UNCONSTRAINED capital k*(z,ell); the collateral constraint bites only in the
#   household block, where an entrepreneur with assets a produces with k_eff and
#   takes the market wage schedule as given. Full GE (theta responding to the
#   entrant wealth distribution) is left for a later step.
#
# NOTE: cf and other calibration values are not yet disciplined to data.
#
# ABILITY PROCESS z (Change 2):
#   Marginal: Pareto on [z_min, inf) with shape alpha_z (fat right tail), discretized
#   onto n_z equal-probability bins (mid-quantile grid points). Legacy Tauchen/log-normal
#   available via use_pareto_z=false.
#   fixed_z=false (default): log-AR(1) persistence on the Pareto grid (rho_z, sigma_z).
#   fixed_z=true: each person draws z once from the Pareto bins and keeps it for life;
#   VFI skips z-expectations; simulation never updates z.
# =============================================================================

using LinearAlgebra, Statistics, Printf, Optim, Plots, Random

# =============================================================================
# SECTION 1: Parameters
# =============================================================================

struct Params
    # preferences
    beta    :: Float64
    sigma   :: Float64      # CRRA
    b       :: Float64      # unemployment benefit / strike threat point

    # production: firm output = z * (k^alpha * ell^(1-alpha))^nu, with nu<1.
    # Capital k is CHOSEN by the entrepreneur subject to k <= lambda*a (Section 2).
    alpha   :: Float64      # capital share within the CRS aggregate
    nu      :: Float64      # span of control / returns to scale (nu < 1)
    delta   :: Float64      # capital depreciation rate
    lambda  :: Float64      # collateral limit: k_eff <= lambda * assets
    r       :: Float64

    # labor market frictions
    s       :: Float64      # separation rate
    d       :: Float64      # firm destruction rate
    eta     :: Float64      # worker bargaining power
    kappa_v :: Float64      # vacancy posting cost
    kappa_e :: Float64      # firm entry cost (free entry pins down theta)
    cf      :: Float64      # fixed per-period operating cost of a firm

    # matching: mu(u,v) = A * u^xi * v^(1-xi)
    A       :: Float64
    xi      :: Float64

    # ability / productivity grid (shared: 2a)
    n_z     :: Int
    z_grid  :: Vector{Float64}
    Pi_z    :: Matrix{Float64}
    pi_z    :: Vector{Float64}
    # ability process: Pareto marginal (fat tail) + optional fixed-for-life z
    use_pareto_z :: Bool     # true = Pareto quantile grid; false = legacy Tauchen/log-normal
    alpha_z   :: Float64     # Pareto shape (lower = fatter right tail)
    z_min     :: Float64     # Pareto scale / lower bound on z
    fixed_z   :: Bool        # true = z drawn once, no transitions; false = Markov persistent
    rho_z     :: Float64     # log-AR(1) persistence (persistent mode only)
    sigma_z   :: Float64     # log-AR(1) shock sd   (persistent mode only)

    # firm size grid
    n_ell   :: Int
    ell_grid :: Vector{Float64}

    # asset grid
    n_a     :: Int
    a_min   :: Float64
    a_max   :: Float64
    a_grid  :: Vector{Float64}
end

function make_params(;
    beta=0.96, sigma=2.0, b=0.40,
    alpha=0.33, nu=0.75, delta=0.06, lambda=1.0, r=0.03,
    s=0.03, d=0.02, eta=0.40, kappa_v=0.30, kappa_e=5.0, cf=0.50,
    A=0.30, xi=0.50,
    n_z=15,
    use_pareto_z=true, alpha_z=1.35, z_min=0.40, fixed_z=false,
    rho_z=0.90, sigma_z=0.20,
    n_ell=30, ell_max=40.0,
    n_a=80, a_min=0.0, a_max=50.0,
)
    z_grid, Pi_z, pi_z = make_ability_process(
        n_z; use_pareto_z, alpha_z, z_min, fixed_z, rho_z, sigma_z)
    ell_grid = collect(range(0.0, ell_max, length=n_ell))   # start at 0 workers
    a_grid = a_min .+ (a_max - a_min) .* (range(0,1,length=n_a)).^2
    return Params(beta, sigma, b, alpha, nu, delta, lambda, r, s, d, eta,
                  kappa_v, kappa_e, cf, A, xi, n_z, z_grid, Pi_z, pi_z,
                  use_pareto_z, alpha_z, z_min, fixed_z, rho_z, sigma_z,
                  n_ell, ell_grid, n_a, a_min, a_max, a_grid)
end

# ---- Ability process: Pareto marginal + optional persistence ----
#
# Pareto type I on [z_min, inf):  F(z) = 1 - (z_min/z)^alpha_z.
# Discretization: equal-probability bins on the existing n_z points.
#   z_i = z_min / (1 - p_i)^(1/alpha_z),  p_i = (i - 0.5)/n_z  (mid-quantile).
#   pi_z = uniform 1/n_z  (exact bin masses).
# Approximation: each bin is summarized by one point (the mid-quantile), not the
# conditional mean within the bin; negligible when n_z is moderate.
#
# Persistent (fixed_z=false): log z follows AR(1) with (rho_z, sigma_z), and
# transitions are computed on the Pareto z_grid (Tauchen-style on log z).
# The Markov chain's invariant pi_z is then iterated from Pi_z; it generally
# DIFFERS slightly from the pure Pareto bin weights — honest caveat.
#
# Fixed for life (fixed_z=true): Pi_z = I; pi_z = Pareto bin weights; no z
# expectations in VFI and no z updates in simulation.

function discretize_pareto(n_z, alpha_z, z_min)
    @assert alpha_z > 0.0 && z_min > 0.0
    z_grid = [z_min / (1 - (i - 0.5)/n_z)^(1/alpha_z) for i in 1:n_z]
    pi_z = fill(1/n_z, n_z)
    return z_grid, pi_z
end

function log_ar1_transition(z_grid, rho, sigma)
    n = length(z_grid)
    logz = log.(z_grid)
    Pi = zeros(n, n)
    for i in 1:n
        mu = rho * logz[i]
        for j in 1:n
            lo = j == 1 ? -Inf : (logz[j-1] + logz[j]) / 2
            hi = j == n ? Inf  : (logz[j] + logz[j+1]) / 2
            Pi[i,j] = ncdf((hi - mu)/sigma) - ncdf((lo - mu)/sigma)
        end
        s = sum(Pi[i,:])
        s > 0.0 && (Pi[i,:] ./= s)
    end
    return Pi
end

function markov_stationary(Pi; max_iter=5000, tol=1e-12)
    n = size(Pi, 1)
    pi = fill(1/n, n)
    for _ in 1:max_iter
        pn = Pi' * pi
        maximum(abs.(pn - pi)) < tol && return pn
        pi = pn
    end
    return pi
end

function make_ability_process(n_z;
    use_pareto_z=true, alpha_z=1.35, z_min=0.40, fixed_z=false,
    rho_z=0.90, sigma_z=0.20)
    if use_pareto_z
        z_grid, pi_pareto = discretize_pareto(n_z, alpha_z, z_min)
        if fixed_z
            Pi_z = Matrix{Float64}(I, n_z, n_z)
            pi_z = pi_pareto
        else
            Pi_z = log_ar1_transition(z_grid, rho_z, sigma_z)
            pi_z = markov_stationary(Pi_z)
        end
    else
        z_g, Pi_z, pi_z = tauchen(n_z, rho_z, sigma_z)
        z_grid = exp.(z_g)
        if fixed_z
            Pi_z = Matrix{Float64}(I, n_z, n_z)
        end
    end
    return z_grid, Pi_z, pi_z
end

# ---- Tauchen (legacy log-normal ability; used when use_pareto_z=false) ----
function tauchen(n, rho, sigma; m=3.0)
    su = sigma/sqrt(1-rho^2)
    zg = collect(range(-m*su, m*su, length=n)); dz = zg[2]-zg[1]
    Pi = zeros(n,n)
    for i in 1:n, j in 1:n
        lo=zg[j]-dz/2; hi=zg[j]+dz/2; mu=rho*zg[i]
        Pi[i,j] = j==1 ? ncdf((hi-mu)/sigma) :
                  j==n ? 1-ncdf((lo-mu)/sigma) :
                         ncdf((hi-mu)/sigma)-ncdf((lo-mu)/sigma)
    end
    for i in 1:n; Pi[i,:] ./= sum(Pi[i,:]); end
    pi = fill(1/n, n)
    for _ in 1:2000
        pn = Pi'*pi; maximum(abs.(pn-pi))<1e-12 && break; pi=pn
    end
    return zg, Pi, pi
end
function erf_as(x)
    sg=sign(x); x=abs(x); t=1/(1+0.3275911x)
    y=1-(((((1.061405429t-1.453152027)t)+1.421413741)t-0.284496736)t+0.254829592)t*exp(-x*x)
    return sg*y
end
ncdf(x) = 0.5*(1+erf_as(x/sqrt(2)))

util(c,p) = c<=1e-10 ? -1e6 : (abs(p.sigma-1)<1e-8 ? log(c) : (c^(1-p.sigma)-1)/(1-p.sigma))


# =============================================================================
# SECTION 2: Matching, Production, Wage  (shared building blocks)
# =============================================================================

f_theta(theta,p) = min(p.A*theta^(1-p.xi), 1.0)    # job-finding rate (prob, capped at 1)
q_theta(theta,p) = min(p.A*theta^(-p.xi), 1.0)     # vacancy-filling rate (prob, capped at 1)

# ---- Capital: user cost, optimal (unconstrained) choice, collateral limit ----
# User cost of capital MPK = r + delta + d : opportunity cost + depreciation +
# the capital lost when the firm is destroyed (prob d).
ucost(p) = p.r + p.delta + p.d

# Output with span of control: Y = z * (k^alpha * ell^(1-alpha))^nu
#                                = z * k^(alpha*nu) * ell^((1-alpha)*nu).
f_output(z,ell,k,p) = (ell<=0.0 || k<=0.0) ? 0.0 :
    z * k^(p.alpha*p.nu) * ell^((1-p.alpha)*p.nu)

# Unconstrained optimal capital, from MPK = u_k:
#   k*(z,ell) = (alpha*nu*z/u_k)^(1/(1-alpha*nu)) * ell^((1-alpha)*nu/(1-alpha*nu)).
function kstar(z,ell,p)
    ell<=0.0 && return 0.0
    an = p.alpha*p.nu
    (an*z/ucost(p))^(1/(1-an)) * ell^(((1-p.alpha)*p.nu)/(1-an))
end

# Collateral-constrained capital: cannot deploy more than lambda * own assets.
k_eff(z,ell,a,p) = min(kstar(z,ell,p), p.lambda*a)

# MPL is the marginal product of labor at the firm's CURRENT capital k, evaluated
# at a floor of the smallest positive firm size (the first worker's MPL is
# unbounded as ell->0, so the bargain is taken at the size a worker actually
# joins). MPL = (1-alpha)*nu * Y / ell.
function MPL(z,ell,k,p)
    ell_eff = max(ell, p.ell_grid[2])
    (1-p.alpha)*p.nu * z * k^(p.alpha*p.nu) * ell_eff^((1-p.alpha)*p.nu - 1)
end

# Firm profit given capital k and wage w (capital charged at its user cost).
pi_firm(z,ell,k,w,p) = f_output(z,ell,k,p) - w*ell - ucost(p)*k

# Nash wage at capital k: threat point b (strike), independent of worker assets.
nash_wage(z,ell,k,theta,p) = p.eta*(MPL(z,ell,k,p)+p.kappa_v*theta) + (1-p.eta)*p.b

# Labor-block convenience wrappers: evaluate at the UNCONSTRAINED optimal capital
# k*(z,ell). These define the market wage schedule and firm profit used to pin
# theta and the firm value (the 5a-preserving approximation; see header).
wage_sched(z,ell,theta,p)   = nash_wage(z,ell,kstar(z,ell,p),theta,p)
pi_firm_unc(z,ell,w,p)      = pi_firm(z,ell,kstar(z,ell,p),w,p)

# interpolation helpers
function interp_ell(ell_grid, V::AbstractVector, ell)
    ell=clamp(ell, ell_grid[1], ell_grid[end])
    i=searchsortedfirst(ell_grid, ell); i=clamp(i,2,length(ell_grid))
    t=(ell-ell_grid[i-1])/(ell_grid[i]-ell_grid[i-1])
    return (1-t)*V[i-1]+t*V[i]
end
function interp_a(V::AbstractVector, p, a)
    a=clamp(a, p.a_grid[1], p.a_grid[end])
    i=searchsortedfirst(p.a_grid, a); i=clamp(i,2,p.n_a)
    t=(a-p.a_grid[i-1])/(p.a_grid[i]-p.a_grid[i-1])
    return (1-t)*V[i-1]+t*V[i]
end


# =============================================================================
# SECTION 3: LABOR-MARKET BLOCK (solve first, 5a)
#
# This is the DMP firm problem. A firm of ability/productivity z with ell
# workers chooses vacancies v to maximize firm value E_firm(z,ell). Firm size
# evolves: ell' = (1-s)*ell + q(theta)*v. Free entry pins down theta.
#
# This block is independent of household wealth because of the strike threat
# point. Output: theta*, the firm value E_firm(z,ell), and vacancy policy.
# =============================================================================

function solve_firm_vfi(theta, p; tol=1e-8, max_iter=2000)
    qt = q_theta(theta,p)
    E = zeros(p.n_z, p.n_ell); v_pol = zeros(p.n_z, p.n_ell)
    for _ in 1:max_iter
        En = similar(E)
        for iz in 1:p.n_z
            z = p.z_grid[iz]
            if p.fixed_z
                Eexp = E[iz,:]
            else
                Eexp = zeros(p.n_ell)
                for iz2 in 1:p.n_z; Eexp .+= p.Pi_z[iz,iz2].*E[iz2,:]; end
            end
            for il in 1:p.n_ell
                ell=p.ell_grid[il]; w=wage_sched(z,ell,theta,p)
                pif = pi_firm_unc(z,ell,w,p)   # unconstrained k* (labor block)
                obj(v)= begin
                    v=max(v,0.0); en=(1-p.s)*ell+qt*v
                    -(pif - p.kappa_v*v + p.beta*(1-p.d)*interp_ell(p.ell_grid,Eexp,en))
                end
                vmax=(p.ell_grid[end]-(1-p.s)*ell)/max(qt,1e-8)+1.0; vmax=max(vmax,0.0)
                vstar = vmax<1e-10 ? 0.0 : max(Optim.minimizer(optimize(obj,0.0,vmax,Brent())),0.0)
                en=(1-p.s)*ell+qt*vstar
                En[iz,il]=pif-p.kappa_v*vstar+p.beta*(1-p.d)*interp_ell(p.ell_grid,Eexp,en)
                v_pol[iz,il]=vstar
            end
        end
        err=maximum(abs.(En.-E)); E.=En; err<tol && break
    end
    return E, v_pol
end

function free_entry_residual(theta,p; ell_entry=0.0)
    E,_ = solve_firm_vfi(theta,p)
    # a new entrant has ell=0 workers; value at entry must equal entry cost
    Ee = sum(p.pi_z[iz]*interp_ell(p.ell_grid, E[iz,:], ell_entry) for iz in 1:p.n_z)
    return Ee - p.kappa_e
end

function bisect(f,lo,hi; tol=1e-6, max_iter=60, verbose=true)
    flo=f(lo); fhi=f(hi)
    @assert flo*fhi<0 "bisect: need opposite signs at bracket ends"
    mid=(lo+hi)/2
    for i in 1:max_iter
        mid=(lo+hi)/2; fm=f(mid)
        verbose && println(@sprintf("  bisect %2d | theta=%.4f | resid=%+.3e", i, mid, fm))
        abs(fm)<tol && break
        flo*fm<0 ? (hi=mid) : (lo=mid)
    end
    return mid
end

function solve_labor_market(p; theta_lo=0.05, theta_hi=20.0)
    println("[1/2] Labor market: solving theta via free entry ...")
    rlo=free_entry_residual(theta_lo,p); rhi=free_entry_residual(theta_hi,p)
    println(@sprintf("      entry residual: theta=%.2f -> %+.4f | theta=%.2f -> %+.4f",
                     theta_lo, rlo, theta_hi, rhi))
    if rlo*rhi>0
        error("no sign change in theta bracket. Entry value vs cost is the same " *
              "sign at both ends; lower kappa_e (entry cost) or check parameters.")
    end
    theta = bisect(th->free_entry_residual(th,p), theta_lo, theta_hi)
    E, v_pol = solve_firm_vfi(theta,p)
    println(@sprintf("      theta* = %.4f,  f(theta*) = %.4f", theta, f_theta(theta,p)))
    return theta, E, v_pol
end


# =============================================================================
# SECTION 4: HOUSEHOLD BLOCK with OCCUPATIONAL CHOICE
#
# Given theta and the firm's vacancy policy v_pol from the labor market, solve
# the household problem. The unemployed/worker states are (assets a, ability z);
# the ENTREPRENEUR state adds the firm's current size ell -> (a, z, ell):
#
#   Vu(a,z):      unemployed -> max( search for a job , start a firm at ell=0 )
#   Vw(a,z):      worker     -> max( keep working , quit to start a firm )
#   Ve(a,z,ell):  entrepreneur running a firm of current size ell
#
# CONSISTENCY WITH THE LABOR BLOCK (the fix):
#   The entrepreneur's per-period payoff is the firm's ACTUAL operating profit
#   pi_firm(z, ell, w(z,ell)) - cf at the firm's current size ell, and the firm
#   grows along the SAME law of motion and SAME vacancy policy v_pol used to
#   compute E_firm in the labor block:  ell' = (1-s)*ell + q(theta)*v_pol(z,ell).
#   A new firm starts at ell=0. So the entrepreneur consumes exactly the profit
#   stream that underlies E_firm(z,0); the household value differs from E_firm
#   only by (i) CRRA utility vs the firm block's risk-neutral PV, (ii) the
#   savings/borrowing margin, and (iii) the voluntary-exit option below. This
#   replaces the old steady-state-profit summary (prof_ss), which charged the
#   entrepreneur the fully-grown profit every period and ignored the start-up
#   phase and mean reversion.
#
# EXIT OPTION: an incumbent entrepreneur may shut the firm down and return to
#   unemployment, Ve = max( operate , Vu ). Without this an entrepreneur is
#   trapped paying cf forever; when z mean-reverts to a low value the firm runs
#   a loss with no way out, which (through negative consumption -> -1e6) poisons
#   the whole entrepreneur value and is what drove the 0%-entrepreneur result.
#
# Workers are still summarized at (a,z): the wage is the Nash wage at a steady-
# size firm of productivity z (decision 1a). Savings a' is chosen by continuous
# optimization subject to a' >= a_min.
# =============================================================================

function solve_household(theta, E_firm, v_pol, p;
                         tol=1e-7, max_iter=2000, verbose=false,
                         n_ell_e=24, ell_max_e=10.0)
    nz, na = p.n_z, p.n_a
    ft = f_theta(theta,p)
    delta = p.s + p.d - p.s*p.d
    qt = q_theta(theta,p)

    # Household firm-size grid for the entrepreneur value Ve(a,z,ell). Firms
    # start at ell=0 and grow toward ell_ss(z) (<= ell_ss of the top z), so the
    # grid spans [0, ell_max_e] with headroom above the largest steady size.
    ell_grid_e = collect(range(0.0, ell_max_e, length=n_ell_e))

    # ell_ss(z), prof_ss(z): the firm's steady size and profit (labor-side
    # objects, kept for reporting and for the worker wage summary).
    ell_ss = zeros(nz); prof_ss = zeros(nz)
    for iz in 1:nz
        ell = 0.0
        for _ in 1:1000
            v = interp_ell(p.ell_grid, v_pol[iz,:], ell)
            en = (1-p.s)*ell + qt*v
            abs(en-ell)<1e-10 && (ell=en; break); ell=en
        end
        ell_ss[iz] = ell
        prof_ss[iz] = pi_firm_unc(p.z_grid[iz], ell, wage_sched(p.z_grid[iz],ell,theta,p), p)
    end

    # worker wage by (z): market wage at the firm's steady size (1a summary)
    wage_z = [wage_sched(p.z_grid[iz], ell_ss[iz], theta, p) for iz in 1:nz]

    # Pre-tabulate entrepreneur operating profit and next firm size. Profit now
    # depends on the OWNER's assets a through the collateral constraint
    # k_eff = min(k*(z,ell), lambda*a): a poorer entrepreneur deploys less
    # capital and earns less. They take the market wage schedule w(z,ell) as
    # given (5a-preserving) and pay the user cost on their k_eff. Next firm size
    # ellp follows the labor block's (unconstrained) vacancy policy, so it is a
    # function of (z,ell) only.
    prof_e = zeros(na, nz, n_ell_e); ellp_e = zeros(nz, n_ell_e)
    for iz in 1:nz
        z = p.z_grid[iz]
        for iel in 1:n_ell_e
            ell = ell_grid_e[iel]
            w = wage_sched(z, ell, theta, p)
            for ia in 1:na
                keff = k_eff(z, ell, p.a_grid[ia], p)
                prof_e[ia,iz,iel] = f_output(z,ell,keff,p) - w*ell - ucost(p)*keff
            end
            v = interp_ell(p.ell_grid, v_pol[iz,:], ell)
            ellp_e[iz,iel] = (1-p.s)*ell + qt*v
        end
    end

    Vu=zeros(na,nz); Vw=zeros(na,nz); Vw_stay=zeros(na,nz)
    Ve=zeros(na,nz,n_ell_e)          # entrepreneur value WITH exit option
    Ve_op=zeros(na,nz,n_ell_e)       # value of OPERATING (no exit) this period
    apU=zeros(na,nz); apW=zeros(na,nz)
    apE=zeros(na,nz,n_ell_e); exitE=falses(na,nz,n_ell_e)
    occ=zeros(Int,na,nz)             # 1=search, 2=start firm (unemployed choice)

    iel0 = 1                         # ell_grid_e[1] == 0.0 (entry size)

    for iter in 1:max_iter
        Vun=copy(Vu); Vwn=copy(Vw); Vwsn=copy(Vw_stay)
        Ven=copy(Ve); Ve_opn=copy(Ve_op)
        for iz in 1:nz
            # z-expectations of the (a,z) value functions (skip when z is fixed for life)
            if p.fixed_z
                EVu = Vu[:,iz]
                EVw = Vw[:,iz]
            else
                EVu=zeros(na); EVw=zeros(na)
                for iz2 in 1:nz
                    EVu .+= p.Pi_z[iz,iz2].*Vu[:,iz2]
                    EVw .+= p.Pi_z[iz,iz2].*Vw[:,iz2]
                end
            end

            # ---- ENTREPRENEUR: operate value Ve_op(a,z,ell) over the ell grid
            for iel in 1:n_ell_e
                ellp = ellp_e[iz,iel]
                # E[ Ve(a', z', ell') ] over z' (degenerate when fixed_z)
                if p.fixed_z
                    EVe_ellp = [interp_ell(ell_grid_e, view(Ve, ia, iz, :), ellp) for ia in 1:na]
                else
                    EVe_ellp=zeros(na)
                    for iz2 in 1:nz
                        w2=p.Pi_z[iz,iz2]; w2==0.0 && continue
                        @inbounds for ia in 1:na
                            EVe_ellp[ia] += w2*interp_ell(ell_grid_e, view(Ve,ia,iz2,:), ellp)
                        end
                    end
                end
                contE(ap)=(1-p.d)*interp_a(EVe_ellp,p,ap)+p.d*interp_a(EVu,p,ap)
                for ia in 1:na
                    a=p.a_grid[ia]
                    resE=(1+p.r)*a + prof_e[ia,iz,iel] - p.cf   # profit uses k_eff(a)
                    aMaxE=min(resE-1e-8, p.a_grid[end])
                    if aMaxE<=p.a_min
                        ap=p.a_min; op=util(resE-ap,p)+p.beta*contE(ap)
                    else
                        rr=optimize(ap->-(util(resE-ap,p)+p.beta*contE(ap)),p.a_min,aMaxE,Brent())
                        ap=Optim.minimizer(rr); op=-Optim.minimum(rr)
                    end
                    apE[ia,iz,iel]=ap; Ve_opn[ia,iz,iel]=op
                end
            end

            for ia in 1:na
                a=p.a_grid[ia]

                # WORKER: earn wage; separate to unemployment at rate delta
                w=wage_z[iz]
                resW=(1+p.r)*a + w
                aMaxW=min(resW-1e-8, p.a_grid[end])
                contW(ap)=(1-delta)*interp_a(EVw,p,ap)+delta*interp_a(EVu,p,ap)
                if aMaxW<=p.a_min
                    ap=p.a_min; ws=util(resW-ap,p)+p.beta*contW(ap); apW[ia,iz]=ap
                else
                    rr=optimize(ap->-(util(resW-ap,p)+p.beta*contW(ap)),p.a_min,aMaxW,Brent())
                    apW[ia,iz]=Optim.minimizer(rr); ws=-Optim.minimum(rr)
                end
                Vwsn[ia,iz]=ws

                # UNEMPLOYED: benefit b; find job w/ prob ft -> worker; else search
                resU=(1+p.r)*a + p.b
                aMaxU=min(resU-1e-8, p.a_grid[end])
                contU(ap)=ft*interp_a(EVw,p,ap)+(1-ft)*interp_a(EVu,p,ap)
                if aMaxU<=p.a_min
                    ap=p.a_min; sv=util(resU-ap,p)+p.beta*contU(ap); apU[ia,iz]=ap
                else
                    rr=optimize(ap->-(util(resU-ap,p)+p.beta*contU(ap)),p.a_min,aMaxU,Brent())
                    apU[ia,iz]=Optim.minimizer(rr); sv=-Optim.minimum(rr)
                end

                # Value of STARTING a firm = operate at entry size ell=0, using
                # the operate value just computed this sweep (entry, quit, and
                # exit all reference the same Ve_op-at-ell=0, so it is consistent).
                start_val = Ve_opn[ia,iz,iel0]

                # occupational choice for the unemployed: search vs start firm
                if start_val>sv
                    Vun[ia,iz]=start_val; occ[ia,iz]=2
                else
                    Vun[ia,iz]=sv; occ[ia,iz]=1
                end
                # worker may quit to start a firm
                Vwn[ia,iz]=max(ws, start_val)
            end
        end

        # exit option: incumbent entrepreneur shuts down -> unemployed (Vu)
        for iz in 1:nz, iel in 1:n_ell_e, ia in 1:na
            if Vun[ia,iz] > Ve_opn[ia,iz,iel]
                Ven[ia,iz,iel]=Vun[ia,iz]; exitE[ia,iz,iel]=true
            else
                Ven[ia,iz,iel]=Ve_opn[ia,iz,iel]; exitE[ia,iz,iel]=false
            end
        end

        err=max(maximum(abs.(Vun-Vu)), maximum(abs.(Vwn-Vw)),
                maximum(abs.(Ve_opn-Ve_op)))
        Vu.=Vun; Vw.=Vwn; Vw_stay.=Vwsn; Ve.=Ven; Ve_op.=Ve_opn
        verbose && iter%25==0 && println(@sprintf("  household VFI %4d | err=%.2e",iter,err))
        err<tol && break
    end

    # 2-D slices at entry size ell=0 (for plots, thresholds, and entry/quit
    # decisions). Ve0 is the value of being an entrepreneur with a brand-new
    # (zero-worker) firm.
    Ve0  = Ve_op[:,:,iel0]
    apE0 = apE[:,:,iel0]
    # worker quits to entrepreneurship where starting a firm beats staying
    quitW = Ve0 .> Vw_stay

    return (Vu=Vu, Vw=Vw, Ve=Ve, Ve_op=Ve_op, Ve0=Ve0, Vw_stay=Vw_stay,
            occ=occ, quitW=quitW, exitE=exitE,
            apU=apU, apW=apW, apE=apE, apE0=apE0,
            ell_grid_e=ell_grid_e, v_pol=v_pol,
            ell_ss=ell_ss, prof_ss=prof_ss, wage_z=wage_z, p=p)
end


# =============================================================================
# SECTION 4b: POPULATION SIMULATION
#
# Simulate N people forward for T periods to get the STATIONARY population
# shares (this is what turns the grid into real population numbers). Each
# person carries assets a and ability z and is in one of three states:
#   1 = unemployed, 2 = worker, 3 = entrepreneur.
#
# Transitions follow the solved policies. Entrepreneurs now carry firm size ell:
#   unemployed: if occ(a,z)=start-firm -> entrepreneur at ell=0; else search,
#               finding a job with prob f(theta) -> worker, else stay unemployed
#   worker:     if quitW(a,z) -> entrepreneur at ell=0; else separate with prob
#               delta -> unemployed, else stay worker
#   entrepreneur: if exitE(a,z,ell) -> shut down, becomes unemployed THIS period
#               (then acts as unemployed); else operate: save apE(a,z,ell), firm
#               grows ell' = (1-s)ell + q(theta) v_pol(z,ell); destroyed w/ prob
#               d -> unemployed (ell reset to 0), else stay entrepreneur at ell'.
# Ability z evolves via the Markov chain each period. Assets update by the
# savings policy of the current occupation.
# =============================================================================

function simulate_population(theta, hh, p; N=20000, T=600, burn=300, seed=1)
    rng = MersenneTwister(seed)
    ft = f_theta(theta,p); delta = p.s + p.d - p.s*p.d; qt = q_theta(theta,p)

    # CDF rows for ability transitions
    cum = cumsum(p.Pi_z, dims=2)
    cum_pi = cumsum(p.pi_z)

    # initial draw
    a = fill(0.0, N)
    z = [searchsortedfirst(cum_pi, rand(rng)) for _ in 1:N]
    state = fill(1, N)   # everyone starts unemployed
    ell = fill(0.0, N)   # firm size; only meaningful while state==3

    nearest_a(x) = clamp(searchsortedfirst(p.a_grid, x), 1, p.n_a)
    nearest_ell(x) = clamp(searchsortedfirst(hh.ell_grid_e, x), 1, length(hh.ell_grid_e))

    share_u=0.0; share_w=0.0; share_e=0.0; nrec=0; ell_sum=0.0; ell_n=0
    k_sum=0.0; kstar_sum=0.0; constr_n=0   # capital & collateral-constraint stats
    for t in 1:T
        for i in 1:N
            ia = nearest_a(a[i]); iz = z[i]
            # entrepreneur who chooses to exit becomes unemployed THIS period,
            # then is handled by the unemployed branch below
            if state[i]==3 && hh.exitE[ia,iz,nearest_ell(ell[i])]
                state[i]=1; ell[i]=0.0
            end

            if state[i]==1            # unemployed
                if hh.occ[ia,iz]==2
                    a[i]=hh.apE0[ia,iz]; ell[i]=0.0; state[i]=3   # start firm
                else
                    a[i]=hh.apU[ia,iz]
                    state[i] = rand(rng)<ft ? 2 : 1
                end
            elseif state[i]==2        # worker
                if hh.quitW[ia,iz]
                    a[i]=hh.apE0[ia,iz]; ell[i]=0.0; state[i]=3   # quit -> firm
                else
                    a[i]=hh.apW[ia,iz]
                    state[i] = rand(rng)<delta ? 1 : 2
                end
            else                      # entrepreneur, operate firm of size ell
                iel = nearest_ell(ell[i])
                a[i]=hh.apE[ia,iz,iel]
                v = interp_ell(p.ell_grid, hh.v_pol[iz,:], ell[i])
                ellnew = (1-p.s)*ell[i] + qt*v
                if rand(rng)<p.d
                    state[i]=1; ell[i]=0.0          # firm destroyed
                else
                    state[i]=3; ell[i]=ellnew
                end
            end
            # ability shock (persistent z only; fixed_z draws z once at initialization)
            if !p.fixed_z
                r=rand(rng); z[i]=searchsortedfirst(cum[iz,:], r)
            end
        end
        if t>burn
            share_u += count(==(1),state)/N
            share_w += count(==(2),state)/N
            share_e += count(==(3),state)/N
            for i in 1:N
                if state[i]==3
                    ell_sum += ell[i]; ell_n += 1
                    ks = kstar(p.z_grid[z[i]], ell[i], p)        # wanted capital
                    ke = min(ks, p.lambda*a[i])                  # deployed capital
                    k_sum += ke; kstar_sum += ks
                    (ks > 0.0 && ke < ks - 1e-9) && (constr_n += 1)
                end
            end
            nrec += 1
        end
    end
    share_u/=nrec; share_w/=nrec; share_e/=nrec
    ell_mean = ell_n>0 ? ell_sum/ell_n : 0.0   # avg firm size among entrepreneurs
    k_mean = ell_n>0 ? k_sum/ell_n : 0.0       # avg capital deployed
    kstar_mean = ell_n>0 ? kstar_sum/ell_n : 0.0
    frac_constrained = ell_n>0 ? constr_n/ell_n : 0.0
    # unemployment rate = unemployed / (unemployed + workers), excludes entrepreneurs
    urate = share_u/(share_u+share_w)
    return (share_u=share_u, share_w=share_w, share_e=share_e, urate=urate,
            ell_mean=ell_mean, k_mean=k_mean, kstar_mean=kstar_mean,
            frac_constrained=frac_constrained,
            assets=copy(a), states=copy(state), ells=copy(ell))
end


# =============================================================================
# SECTION 5: Solve + Report + Plots
# =============================================================================

function solve_model(p)
    println("="^64); println("MERGED MODEL: DMP + BHA + Occupational Choice"); println("="^64)
    ztag = p.fixed_z ? "fixed for life" : "persistent"
    ptag = p.use_pareto_z ? @sprintf("Pareto alpha_z=%.2f", p.alpha_z) : "Tauchen/log-normal"
    println(@sprintf("  Ability: %s (%s)", ptag, ztag))
    theta, E_firm, v_pol = solve_labor_market(p)
    println("\n[2/2] Household block with occupational choice ...")
    hh = solve_household(theta, E_firm, v_pol, p; verbose=true)
    println("\nSimulating population for stationary shares ...")
    sim = simulate_population(theta, hh, p)
    return (theta=theta, E_firm=E_firm, v_pol=v_pol, hh=hh, sim=sim, p=p)
end

function report_model(sol)
    p=sol.p; hh=sol.hh; sim=sol.sim
    println("\n"*"="^64); println("RESULTS"); println("="^64)
    println(@sprintf("\n  theta* = %.4f   f(theta*) = %.4f", sol.theta, f_theta(sol.theta,p)))
    println("\n  POPULATION SHARES (from simulation, these are the real numbers):")
    println(@sprintf("    workers       : %.1f%%", 100*sim.share_w))
    println(@sprintf("    entrepreneurs : %.1f%%", 100*sim.share_e))
    println(@sprintf("    unemployed    : %.1f%%", 100*sim.share_u))
    println(@sprintf("    unemployment rate (u / (u+w)) : %.1f%%", 100*sim.urate))
    println(@sprintf("    avg firm size among entrepreneurs : %.2f workers", sim.ell_mean))
    println("\n  CAPITAL & COLLATERAL CONSTRAINT (k_eff = min(k*, lambda*a), lambda=$(p.lambda)):")
    println(@sprintf("    avg capital deployed (k_eff)         : %.2f", sim.k_mean))
    println(@sprintf("    avg desired capital (k*)             : %.2f", sim.kstar_mean))
    println(@sprintf("    entrepreneurs hitting the constraint : %.1f%%", 100*sim.frac_constrained))
    frac=count(==(2), hh.occ)/length(hh.occ)
    println(@sprintf("\n  (For reference: entrepreneurship covers %.1f%% of the (a,z) GRID,", 100*frac))
    println( "   which is NOT the population share above — the grid is unweighted.)")
    println("\n  Entry threshold: min assets to start a firm, by ability z.")
    println("  k*(z) is the unconstrained optimal capital at the steady firm size;")
    println("  if the entry threshold tracks k*, the constraint is what locks people out.")
    println(@sprintf("  %-10s %-12s %-12s %-12s","z","min assets","ss firm size","k*(z,ss)"))
    println("  "*"-"^48)
    for iz in 1:p.n_z
        ia=findfirst(==(2), hh.occ[:,iz])
        cut = ia===nothing ? "never" : @sprintf("%.3f", p.a_grid[ia])
        ks = kstar(p.z_grid[iz], hh.ell_ss[iz], p)
        println(@sprintf("  %-10.4f %-12s %-12.2f %-12.2f", p.z_grid[iz], cut, hh.ell_ss[iz], ks))
    end
end

function plot_model(sol; outdir=".")
    p=sol.p; hh=sol.hh

    # Occupation map. Colors are pinned so 1=worker and 2=entrepreneur always
    # map to the same two colors even if only one occupation is present (a
    # single-value heatmap would otherwise stretch its scale and mislead).
    plt1=heatmap(p.a_grid, p.z_grid, hh.occ';
        xlabel="assets a", ylabel="ability z",
        title="Occupational Choice",
        c=cgrad([:steelblue, :gold], 2, categorical=true),
        clims=(1,2), colorbar=false)
    # legend via dummy series
    scatter!(plt1, [NaN],[NaN]; m=:square, mc=:steelblue, label="worker")
    scatter!(plt1, [NaN],[NaN]; m=:square, mc=:gold, label="entrepreneur")
    plot!(plt1; legend=:topright)
    savefig(plt1, joinpath(outdir,"merged_occupation_map.png"))

    # value of staying a worker vs starting a firm (entrepreneur value at the
    # entry firm size ell=0), at a high ability where entry is relevant
    iz=(p.n_z+1)÷2; rng=2:p.n_a
    wl=hh.Vw_stay[rng,iz]; el=hh.Ve0[rng,iz]; best=max.(wl,el)
    plt2=plot(p.a_grid[rng], wl; label="worker", lw=2, xlabel="assets a", ylabel="value",
        title="Value by Occupation (median ability z)", ylims=(-10, maximum(best)+5))
    plot!(plt2, p.a_grid[rng], el; label="entrepreneur (start a firm)", lw=2)
    plot!(plt2, p.a_grid[rng], best; label="best choice", lw=2, ls=:dash, lc=:black)
    savefig(plt2, joinpath(outdir,"merged_occupation_values.png"))

    # savings policies at median ability (entrepreneur shown at entry size ell=0)
    plt3=plot(p.a_grid, hh.apW[:,iz]; label="worker", lw=2, xlabel="assets a",
        ylabel="next assets a'", title="Savings Policy (median ability z)")
    plot!(plt3, p.a_grid, hh.apE0[:,iz]; label="entrepreneur (ell=0)", lw=2)
    plot!(plt3, p.a_grid, hh.apU[:,iz]; label="unemployed", lw=2)
    plot!(plt3, p.a_grid, p.a_grid; label="45 deg", ls=:dash, lc=:gray)
    savefig(plt3, joinpath(outdir,"merged_savings_policy.png"))

    # firm size by ability
    plt4=plot(p.z_grid, hh.ell_ss; lw=2, legend=false, xlabel="ability z",
        ylabel="steady firm size ell", title="Firm Size by Ability")
    savefig(plt4, joinpath(outdir,"merged_firm_size.png"))

    # wealth distribution from the simulation
    plt5=histogram(sol.sim.assets; bins=50, legend=false, xlabel="assets a",
        ylabel="number of people", title="Stationary Wealth Distribution")
    savefig(plt5, joinpath(outdir,"merged_wealth_distribution.png"))

    println("\nSaved 5 plots to: $(abspath(outdir))")
end

function main()
    p=make_params()
    sol=solve_model(p)
    report_model(sol)
    plot_model(sol)
    return sol
end

# Run the full pipeline only when executed directly (`julia Merged_Model.jl`),
# not when `include`d by an analysis script (e.g. the lambda sweep).
if abspath(PROGRAM_FILE) == @__FILE__
    sol = main()
end
nothing