# =============================================================================
# Financial-development comparative static: sweep the collateral limit lambda
# (k_eff = min(k*, lambda*a)) and trace how the entrepreneurial ENTRY THRESHOLD
# and the CAPITAL-CONSTRAINED share move with it. Higher lambda = more developed
# financial system (more borrowing against collateral).
#
# The labor block (theta, wage schedule, firm value, hiring policy) is computed
# at UNCONSTRAINED capital and so does NOT depend on lambda -> solve it ONCE and
# reuse it for every lambda; only the household block + simulation are redone.
# =============================================================================
include("Merged_Model.jl")

lams = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

p0 = make_params(nu=0.75)
println("Solving labor block once (lambda-independent) ...")
theta, E, v_pol = solve_labor_market(p0)

# abilities that actually run firms (the top two in this calibration)
iz_top = p0.n_z       # z = 3.96
iz_mid = p0.n_z - 1   # z = 2.50

thr_top   = fill(NaN, length(lams))   # min assets to start a firm, top ability
thr_mid   = fill(NaN, length(lams))   # min assets to start a firm, mid-high ability
frac_con  = zeros(length(lams))       # share of entrepreneurs hitting k_eff < k*
share_e   = zeros(length(lams))       # entrepreneur share of population
kgap      = zeros(length(lams))       # avg (k* - k_eff)/k* : capital shortfall

io = open("/tmp/sweep_out.txt", "w")
println(io, "lambda  thr(z=3.96)  thr(z=2.50)  ent_share%  constrained%  cap_shortfall%")
for (i, lam) in enumerate(lams)
    p = make_params(nu=0.75, lambda=lam)
    hh = solve_household(theta, E, v_pol, p)
    sim = simulate_population(theta, hh, p)

    iat = findfirst(==(2), hh.occ[:, iz_top]); thr_top[i] = iat===nothing ? NaN : p.a_grid[iat]
    iam = findfirst(==(2), hh.occ[:, iz_mid]); thr_mid[i] = iam===nothing ? NaN : p.a_grid[iam]
    frac_con[i] = sim.frac_constrained
    share_e[i]  = sim.share_e
    kgap[i]     = sim.kstar_mean>0 ? (sim.kstar_mean - sim.k_mean)/sim.kstar_mean : 0.0

    @printf(io, "%6.2f  %10.3f  %10.3f  %9.2f  %11.2f  %12.2f\n",
        lam, thr_top[i], thr_mid[i], 100*share_e[i], 100*frac_con[i], 100*kgap[i])
    flush(io)
    @printf("  lambda=%.2f done: ent=%.1f%%, constrained=%.1f%%, thr(top)=%.2f\n",
        lam, 100*share_e[i], 100*frac_con[i], thr_top[i])
end
close(io)

# ---- figure: entry threshold (left) and constrained share (right) vs lambda ----
plt1 = plot(lams, thr_top; marker=:circle, lw=2, label="z = 3.96 (top ability)",
    xlabel="λ  (collateral limit, k ≤ λ·a)", ylabel="min assets to start a firm",
    title="Entry threshold vs financial development", xscale=:log10, legend=:topright)
plot!(plt1, lams, thr_mid; marker=:square, lw=2, label="z = 2.50")

plt2 = plot(lams, 100 .* frac_con; marker=:circle, lw=2, label="capital-constrained entrepreneurs",
    xlabel="λ  (collateral limit, k ≤ λ·a)", ylabel="percent",
    title="Constraint incidence vs financial development", xscale=:log10, legend=:topright)
plot!(plt2, lams, 100 .* kgap; marker=:diamond, lw=2, ls=:dash, label="avg capital shortfall (k*−k_eff)/k*")
plot!(plt2, lams, 100 .* share_e; marker=:square, lw=2, label="entrepreneur share of population")

fig = plot(plt1, plt2; layout=(1,2), size=(1050,430), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
savefig(fig, "merged_lambda_sweep.png")
println("\nSaved figure: ", abspath("merged_lambda_sweep.png"))
println("DONE_MARKER")
