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
# NOTE: still uses a fixed operating cost cf and span control via the DMP
# size mechanism. cf and other calibration values are not yet disciplined to
# data.
# =============================================================================

using LinearAlgebra, Statistics, Printf, Optim, Plots

# =============================================================================
# SECTION 1: Parameters
# =============================================================================

struct Params
    # preferences
    beta    :: Float64
    sigma   :: Float64      # CRRA
    b       :: Float64      # unemployment benefit / strike threat point

    # production: firm output = z * k^alpha * ell^(1-alpha)
    alpha   :: Float64
    k       :: Float64      # capital per firm (fixed for now)
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
    alpha=0.33, k=1.0, r=0.03,
    s=0.03, d=0.02, eta=0.50, kappa_v=0.50, kappa_e=1.0, cf=0.50,
    A=0.70, xi=0.50,
    n_z=7, rho_z=0.90, sigma_z=0.20,
    n_ell=30, ell_max=15.0,
    n_a=80, a_min=0.0, a_max=50.0,
)
    z_grid, Pi_z, pi_z = tauchen(n_z, rho_z, sigma_z)
    z_grid = exp.(z_grid)
    ell_grid = collect(range(0.0, ell_max, length=n_ell))   # start at 0 workers
    a_grid = a_min .+ (a_max - a_min) .* (range(0,1,length=n_a)).^2
    return Params(beta, sigma, b, alpha, k, r, s, d, eta, kappa_v, kappa_e, cf,
                  A, xi, n_z, z_grid, Pi_z, pi_z, n_ell, ell_grid,
                  n_a, a_min, a_max, a_grid)
end

# ---- Tauchen ----
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

f_theta(theta,p) = p.A*theta^(1-p.xi)    # job-finding rate (DERIVED, not a parameter)
q_theta(theta,p) = p.A*theta^(-p.xi)     # vacancy-filling rate

f_prod(z,ell,p) = ell<=0.0 ? 0.0 : z*p.k^p.alpha*ell^(1-p.alpha)
# MPL is evaluated at a floor of the smallest positive firm size: under
# Cobb-Douglas the first worker has unbounded marginal product, so the wage
# bargain must be taken at the size a worker actually joins, not at ell=0.
MPL(z,ell,p)    = begin
    ell_eff = max(ell, p.ell_grid[2])
    (1-p.alpha)*z*p.k^p.alpha*ell_eff^(-p.alpha)
end
pi_firm(z,ell,w,p) = f_prod(z,ell,p) - w*ell - p.r*p.k

# Nash wage: threat point b (strike), so it is independent of worker assets (5a)
nash_wage(z,ell,theta,p) = p.eta*(MPL(z,ell,p)+p.kappa_v*theta) + (1-p.eta)*p.b

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
            Eexp = zeros(p.n_ell)
            for iz2 in 1:p.n_z; Eexp .+= p.Pi_z[iz,iz2].*E[iz2,:]; end
            for il in 1:p.n_ell
                ell=p.ell_grid[il]; w=nash_wage(z,ell,theta,p)
                pif = pi_firm(z,ell,w,p)
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

function solve_labor_market(p; theta_lo=0.05, theta_hi=8.0)
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
# Given theta, the firm value E_firm(z,ell), and the vacancy policy from the
# labor market, solve the household problem over (assets a, ability z):
#
#   Vu(a,z): unemployed -> max( search for a job , start a firm )
#   Vw(a,z): employed worker -> max( keep working , quit to start a firm )
#   Ve(a,z): entrepreneur -> runs a firm (starts at ell=0, grows via hiring)
#
# A worker earns the Nash wage at the firm they're in. To keep the state at
# (a, z) per decision 1a, the worker's wage is summarized by w(z, ell) at the
# firm they belong to; an entrepreneur's value already integrates the firm's
# growth path via E_firm. The entrepreneur's flow payoff each period is the
# firm's profit net of the fixed operating cost cf.
#
# Savings: every occupation chooses a' by continuous optimization, with the
# borrowing constraint a' >= a_min.
# =============================================================================

function solve_household(theta, E_firm, v_pol, p; tol=1e-7, max_iter=2000, verbose=false)
    nz, na = p.n_z, p.n_a
    ft = f_theta(theta,p)
    delta = p.s + p.d - p.s*p.d

    # Entrepreneur per-period profit flow by (z): a firm grows from ell=0, but
    # for the household value we use the firm's average operating profit at its
    # optimal size path, summarized by the steady firm size for productivity z.
    # We take the firm's profit at the size it converges to under v_pol.
    # ell_ss(z): iterate ell' = (1-s)*ell + q(theta)*v*(z,ell) to a fixed point.
    qt = q_theta(theta,p)
    ell_ss = zeros(nz); prof_ss = zeros(nz)
    for iz in 1:nz
        ell = 0.0
        for _ in 1:1000
            v = interp_ell(p.ell_grid, v_pol[iz,:], ell)
            en = (1-p.s)*ell + qt*v
            abs(en-ell)<1e-10 && (ell=en; break); ell=en
        end
        ell_ss[iz] = ell
        w = nash_wage(p.z_grid[iz], ell, theta, p)
        prof_ss[iz] = pi_firm(p.z_grid[iz], ell, w, p)   # firm profit at steady size
    end

    # worker wage by (z): wage at the firm's steady size (1a summary)
    wage_z = [nash_wage(p.z_grid[iz], ell_ss[iz], theta, p) for iz in 1:nz]

    Vu=zeros(na,nz); Vw=zeros(na,nz); Ve=zeros(na,nz); Vw_stay=zeros(na,nz)
    apU=zeros(na,nz); apW=zeros(na,nz); apE=zeros(na,nz); occ=zeros(Int,na,nz)

    for iter in 1:max_iter
        Vun=similar(Vu); Vwn=similar(Vw); Ven=similar(Ve); Vwsn=similar(Vw_stay)
        for iz in 1:nz
            EVu=zeros(na); EVw=zeros(na); EVe=zeros(na)
            for iz2 in 1:nz
                EVu .+= p.Pi_z[iz,iz2].*Vu[:,iz2]
                EVw .+= p.Pi_z[iz,iz2].*Vw[:,iz2]
                EVe .+= p.Pi_z[iz,iz2].*Ve[:,iz2]
            end
            w = wage_z[iz]; prof = prof_ss[iz]
            for ia in 1:na
                a=p.a_grid[ia]

                # ENTREPRENEUR: flow = profit - cf; destroyed -> unemployed
                resE=(1+p.r)*a + prof - p.cf
                aMaxE=min(resE-1e-8, p.a_grid[end])
                contE(ap)=(1-p.d)*interp_a(EVe,p,ap)+p.d*interp_a(EVu,p,ap)
                if aMaxE<=p.a_min
                    ap=p.a_min; Veia=util(resE-ap,p)+p.beta*contE(ap); apE[ia,iz]=ap
                else
                    rr=optimize(ap->-(util(resE-ap,p)+p.beta*contE(ap)),p.a_min,aMaxE,Brent())
                    apE[ia,iz]=Optim.minimizer(rr); Veia=-Optim.minimum(rr)
                end
                Ven[ia,iz]=Veia

                # WORKER: earn wage; separate to unemployment at rate delta;
                # may quit to entrepreneurship
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
                Vwn[ia,iz]=max(ws, Veia)     # quit option

                # UNEMPLOYED: benefit b; find job w/ prob ft -> worker; else search.
                # Occupational choice: search vs start a firm.
                resU=(1+p.r)*a + p.b
                aMaxU=min(resU-1e-8, p.a_grid[end])
                contU(ap)=ft*interp_a(EVw,p,ap)+(1-ft)*interp_a(EVu,p,ap)
                if aMaxU<=p.a_min
                    ap=p.a_min; sv=util(resU-ap,p)+p.beta*contU(ap); apU[ia,iz]=ap
                else
                    rr=optimize(ap->-(util(resU-ap,p)+p.beta*contU(ap)),p.a_min,aMaxU,Brent())
                    apU[ia,iz]=Optim.minimizer(rr); sv=-Optim.minimum(rr)
                end
                if Veia>sv
                    Vun[ia,iz]=Veia; occ[ia,iz]=2
                else
                    Vun[ia,iz]=sv; occ[ia,iz]=1
                end
            end
        end
        err=max(maximum(abs.(Vun-Vu)),maximum(abs.(Vwn-Vw)),maximum(abs.(Ven-Ve)))
        Vu.=Vun; Vw.=Vwn; Ve.=Ven; Vw_stay.=Vwsn
        verbose && iter%25==0 && println(@sprintf("  household VFI %4d | err=%.2e",iter,err))
        err<tol && break
    end

    return (Vu=Vu, Vw=Vw, Ve=Ve, Vw_stay=Vw_stay, occ=occ,
            apU=apU, apW=apW, apE=apE,
            ell_ss=ell_ss, prof_ss=prof_ss, wage_z=wage_z, p=p)
end


# =============================================================================
# SECTION 5: Solve + Report + Plots
# =============================================================================

function solve_model(p)
    println("="^64); println("MERGED MODEL: DMP + BHA + Occupational Choice"); println("="^64)
    theta, E_firm, v_pol = solve_labor_market(p)
    println("\n[2/2] Household block with occupational choice ...")
    hh = solve_household(theta, E_firm, v_pol, p; verbose=true)
    return (theta=theta, E_firm=E_firm, v_pol=v_pol, hh=hh, p=p)
end

function report_model(sol)
    p=sol.p; hh=sol.hh
    println("\n"*"="^64); println("RESULTS"); println("="^64)
    println(@sprintf("\n  theta* = %.4f   f(theta*) = %.4f", sol.theta, f_theta(sol.theta,p)))
    frac=count(==(2), hh.occ)/length(hh.occ)
    println(@sprintf("  Entrepreneurship chosen on %.1f%% of the (a,z) grid", 100*frac))
    println("\n  Entry threshold: min assets to start a firm, by ability z")
    println(@sprintf("  %-12s %-14s %-10s","z","min assets","firm size"))
    println("  "*"-"^36)
    for iz in 1:p.n_z
        ia=findfirst(==(2), hh.occ[:,iz])
        cut = ia===nothing ? "never" : @sprintf("%.3f", p.a_grid[ia])
        println(@sprintf("  %-12.4f %-14s %-10.2f", p.z_grid[iz], cut, hh.ell_ss[iz]))
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

    iz=(p.n_z+1)÷2; rng=2:p.n_a
    wl=hh.Vw_stay[rng,iz]; el=hh.Ve[rng,iz]; best=max.(wl,el)
    plt2=plot(p.a_grid[rng], wl; label="worker", lw=2, xlabel="assets a", ylabel="value",
        title="Value by Occupation (median ability z)", ylims=(-10, maximum(best)+5))
    plot!(plt2, p.a_grid[rng], el; label="entrepreneur", lw=2)
    plot!(plt2, p.a_grid[rng], best; label="best choice", lw=2, ls=:dash, lc=:black)
    savefig(plt2, joinpath(outdir,"merged_occupation_values.png"))

    # savings policies at median ability
    plt3=plot(p.a_grid, hh.apW[:,iz]; label="worker", lw=2, xlabel="assets a",
        ylabel="next assets a'", title="Savings Policy (median ability z)")
    plot!(plt3, p.a_grid, hh.apE[:,iz]; label="entrepreneur", lw=2)
    plot!(plt3, p.a_grid, hh.apU[:,iz]; label="unemployed", lw=2)
    plot!(plt3, p.a_grid, p.a_grid; label="45 deg", ls=:dash, lc=:gray)
    savefig(plt3, joinpath(outdir,"merged_savings_policy.png"))

    # firm size by ability
    plt4=plot(p.z_grid, hh.ell_ss; lw=2, legend=false, xlabel="ability z",
        ylabel="steady firm size ell", title="Firm Size by Ability")
    savefig(plt4, joinpath(outdir,"merged_firm_size.png"))

    println("\nSaved 4 plots to: $(abspath(outdir))")
end

function main()
    p=make_params()
    sol=solve_model(p)
    report_model(sol)
    plot_model(sol)
    return sol
end

sol=main();
nothing