# Compare fixed-for-life vs persistent ability (Pareto z), and collateral effect on headcount.
include("Merged_Model.jl")

function entry_thresholds(hh, p)
    cuts = Vector{Union{String,Float64}}(undef, p.n_z)
    for iz in 1:p.n_z
        ia = findfirst(==(2), hh.occ[:, iz])
        cuts[iz] = ia === nothing ? "never" : p.a_grid[ia]
    end
    return cuts
end

function run_case(label, p; theta=nothing, E=nothing, v_pol=nothing)
    if theta === nothing
        theta, E, v_pol = solve_labor_market(p)
    end
    hh = solve_household(theta, E, v_pol, p)
    sim = simulate_population(theta, hh, p)
    cuts = entry_thresholds(hh, p)
    return (label=label, hh=hh, sim=sim, cuts=cuts, theta=theta, E=E, v_pol=v_pol)
end

lams = [0.5, 1.0]
configs = [("persistent", false), ("fixed_z", true)]

p0 = make_params()
println("z_grid: ", round.(p0.z_grid; digits=3))
println("Solving labor block once ...")
theta, E, v_pol = solve_labor_market(p0)

results = Dict{String,Dict{Float64,NamedTuple}}()
for (name, fixed) in configs
    results[name] = Dict{Float64,NamedTuple}()
    for lam in lams
        p = make_params(lambda=lam, fixed_z=fixed)
        r = run_case("$name λ=$lam", p; theta=theta, E=E, v_pol=v_pol)
        results[name][lam] = r
        @printf("  [%s λ=%.1f] ent=%.2f%%  constrained=%.1f%%  urate=%.1f%%\n",
            name, lam, 100*r.sim.share_e, 100*r.sim.frac_constrained, 100*r.sim.urate)
    end
end

io = open("/tmp/fixed_z_compare.txt", "w")
println(io, "FIXED vs PERSISTENT ABILITY (Pareto z)")
println(io, @sprintf("alpha_z=%.2f  z_min=%.2f  n_z=%d", p0.alpha_z, p0.z_min, p0.n_z))
println(io, "z_grid: ", join(round.(p0.z_grid; digits=3), ", "))
println(io)
println(io, @sprintf("%-12s %6s %10s %12s %12s %12s", "mode", "lambda", "ent_share%", "constr%", "thr_top", "thr_mid"))
println(io, "-"^72)

iz_top = p0.n_z
iz_mid = p0.n_z - 1

for (name, _) in configs
    for lam in lams
        r = results[name][lam]
        thr_top = r.cuts[iz_top]
        thr_mid = r.cuts[iz_mid]
        thr_top_s = thr_top isa String ? thr_top : @sprintf("%.3f", thr_top)
        thr_mid_s = thr_mid isa String ? thr_mid : @sprintf("%.3f", thr_mid)
        @printf(io, "%-12s %6.2f %10.2f %12.2f %12s %12s\n",
            name, lam, 100*r.sim.share_e, 100*r.sim.frac_constrained, thr_top_s, thr_mid_s)
    end
end

println(io)
println(io, "Entry thresholds by z (lambda=1.0):")
println(io, @sprintf("  %-10s %-12s %-12s", "z", "persistent", "fixed_z"))
for iz in 1:p0.n_z
    cp = results["persistent"][1.0].cuts[iz]
    cf = results["fixed_z"][1.0].cuts[iz]
    sp = cp isa String ? cp : @sprintf("%.3f", cp)
    sf = cf isa String ? cf : @sprintf("%.3f", cf)
    @printf(io, "  %-10.3f %-12s %-12s\n", p0.z_grid[iz], sp, sf)
end

println(io)
println(io, "Collateral headcount effect (entrepreneur share change, tight vs loose lambda):")
for (name, _) in configs
    de = 100 * (results[name][0.5].sim.share_e - results[name][1.0].sim.share_e)
    @printf(io, "  %-12s  share_e(λ=0.5) - share_e(λ=1.0) = %+.2f pp\n", name, de)
end
close(io)

println("\nWrote ", abspath("/tmp/fixed_z_compare.txt"))
println("DONE_MARKER")
