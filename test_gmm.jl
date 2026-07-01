# One-shot GMM objective test at default parameters (no optimizer).
include("gmm_calibration.jl")

println("="^78)
println("GMM SCAFFOLDING TEST — single objective evaluation at default parameters")
println("="^78)
println("\nEstimated parameters (fixed everything else at make_params defaults):")
for (i, nm) in enumerate(ESTIMATED_PARAM_NAMES)
    @printf("  %-12s = %.4f\n", String(nm), default_estimated_vector()[i])
end

θ0 = default_estimated_vector()
println("\nSolving model + computing moments (expect ~8 min) ...")
t0 = time()
result = gmm_objective(θ0)
elapsed = time() - t0

print_moment_comparison(result.moments)
@printf("\n  GMM objective (identity W): %.6f\n", result.objective)
@printf("  Elapsed: %.1f min\n", elapsed / 60)

m = result.moments
println("\n  Diagnostics:")
@printf("    implied 1/d (exog exit only)     = %.2f periods\n", m._implied_firm_duration_1_over_d)
@printf("    completed spells (post-burn start): u=%d  w=%d  e=%d  firms=%d\n",
    m._n_u_spells, m._n_w_spells, m._n_e_spells, m._n_firms)
@printf("    right-censored open spells at T : u=%d  w=%d  e=%d  (dropped; should be tiny)\n",
    m._cens_u, m._cens_w, m._cens_e)

println("\n  Wealth/income decomposition:")
@printf("    mean assets (wealth)             = %.4f\n", m._mean_wealth)
@printf("    mean labor/business income       = %.4f\n", m._mean_labor_income)
@printf("    mean capital income (r*a)        = %.4f\n", m._mean_cap_income)
@printf("    mean TOTAL period income         = %.4f\n", m._mean_tot_income)
@printf("    wealth / total income (REPORTED) = %.4f\n", m.wealth_to_income_ratio)
@printf("    wealth / labor income (OLD bug)  = %.4f\n", m._wealth_to_labor_income_ratio)

println("\n  Firm-size dispersion variants:")
@printf("    std across firms, avg-over-life (REPORTED) = %.4f\n", m.std_firm_size)
@printf("    std across firms, size-at-end-of-life      = %.4f\n", m._std_firm_size_end)
@printf("    std over entrepreneur-periods (OLD)        = %.4f\n", m._std_firm_size_period)
@printf("    mean firm size, period-based (REPORTED)    = %.4f\n", m.mean_firm_size)
@printf("    mean firm size, per-firm avg-over-life     = %.4f\n", m._mean_firm_size_perfirm)

println("\n  Moment difficulty notes:")
for nm in MOMENT_NAMES
    level, note = MOMENT_DIFFICULTY[nm]
    println("    $(nm): [$level] $note")
end

print_estimation_readiness()
println("\nDONE_MARKER")
