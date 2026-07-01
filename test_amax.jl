# Test whether the asset-grid ceiling a_max compresses the wealth distribution.
# ONLY a_max varies; everything else is at make_params defaults. The labor block
# (theta, firm value, vacancy policy) does NOT depend on a_max, so it is solved
# ONCE and reused; only the household block + simulation are redone per a_max.
include("gmm_calibration.jl")
using Statistics

amax_values = [50.0, 150.0, 300.0]

println("Solving labor block once (a_max-independent) ...")
p0 = make_params()
theta, E, v_pol = solve_labor_market(p0)
@printf("  theta* = %.4f\n", theta)

results = NamedTuple[]
for amax in amax_values
    @printf("\n>>> a_max = %.0f : solving household + simulating ...\n", amax)
    t0 = time()
    p = make_params(a_max=amax)
    hh = solve_household(theta, E, v_pol, p; verbose=false)
    gsim = simulate_for_gmm(theta, hh, p)
    sol = (theta=theta, hh=hh, p=p)
    moms = model_moments(sol; gmm_sim=gsim)
    d = moment_vector(moms) .- target_vector()
    obj = sum(d .^ 2)
    w = gsim.wealth_obs
    res = (amax=amax, moms=moms, obj=obj,
           wmean=mean(w), wmax=maximum(w),
           w99=quantile(w, 0.99), w50=median(w),
           frac_near_cap=count(>=(0.99 * amax), w) / length(w),
           elapsed=time() - t0)
    push!(results, res)
    @printf("    done in %.1f min | mean wealth=%.3f | wealth/income=%.3f | %% near cap=%.2f%%\n",
        res.elapsed / 60, res.wmean, moms.wealth_to_income_ratio, 100 * res.frac_near_cap)
end

# ----- comparison table -----
io = open("/tmp/amax_test.txt", "w")
function emit(s); println(s); println(io, s); end

emit("\n" * "="^86)
emit("a_max SENSITIVITY  (only a_max varies; all else at defaults, n_a=80, quadratic grid)")
emit("="^86)

cols = [@sprintf("a_max=%.0f", r.amax) for r in results]
emit(@sprintf("  %-38s %14s %14s %14s", "quantity", cols[1], cols[2], cols[3]))
emit("  " * "-"^82)

rowf(name, f) = emit(@sprintf("  %-38s %14.4f %14.4f %14.4f", name,
    f(results[1]), f(results[2]), f(results[3])))

emit("  --- wealth distribution -----------------------------------------------------------")
rowf("mean wealth",                 r -> r.wmean)
rowf("median wealth",               r -> r.w50)
rowf("99th pctile wealth",          r -> r.w99)
rowf("max wealth observed",         r -> r.wmax)
rowf("% population within 1% of cap", r -> 100 * r.frac_near_cap)
emit("  --- the moment under test ---------------------------------------------------------")
rowf("wealth_to_income_ratio",      r -> r.moms.wealth_to_income_ratio)
rowf("mean TOTAL period income",    r -> r.moms._mean_tot_income)
emit("  --- shares (should be ~stable) ----------------------------------------------------")
rowf("entrepreneur_rate",           r -> r.moms.entrepreneur_rate)
rowf("unemployment_rate",           r -> r.moms.unemployment_rate)
emit("  --- other moments (watch for material moves) --------------------------------------")
rowf("mean_firm_duration",          r -> r.moms.mean_firm_duration)
rowf("mean_unemployment_duration",  r -> r.moms.mean_unemployment_duration)
rowf("mean_job_tenure",             r -> r.moms.mean_job_tenure)
rowf("unemp_to_wage_income_ratio",  r -> r.moms.unemp_to_wage_income_ratio)
rowf("worker_to_entrep_income_ratio", r -> r.moms.worker_to_entrepreneur_income_ratio)
rowf("mean_firm_size",              r -> r.moms.mean_firm_size)
rowf("std_firm_size",               r -> r.moms.std_firm_size)
emit("  --- fit ---------------------------------------------------------------------------")
rowf("GMM objective (identity W)",  r -> r.obj)

emit("")
emit("  Interpretation guide:")
emit("   - If mean/99th-pctile wealth keep climbing with a_max and %-near-cap stays high,")
emit("     the ceiling was binding and compressing the distribution (grid artifact).")
emit("   - If they settle and %-near-cap -> ~0, the distribution is interior and the high")
emit("     wealth/income ratio is a genuine model feature, not a grid artifact.")
close(io)
println("\nWrote /tmp/amax_test.txt")
println("DONE_MARKER")
