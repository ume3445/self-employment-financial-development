# Check ability-grid resolution: entry thresholds and entrepreneur share vs n_z.
include("Merged_Model.jl")

function entry_table(hh, p)
    rows = Vector{NamedTuple{(:z, :min_a, :ell_ss, :kstar),Tuple{Float64,Union{String,Float64},Float64,Float64}}}()
    for iz in 1:p.n_z
        ia = findfirst(==(2), hh.occ[:, iz])
        cut = ia === nothing ? "never" : p.a_grid[ia]
        ks = kstar(p.z_grid[iz], hh.ell_ss[iz], p)
        push!(rows, (z=p.z_grid[iz], min_a=cut, ell_ss=hh.ell_ss[iz], kstar=ks))
    end
    return rows
end

function run_nz(n_z; lambda=1.0, fixed_z=false)
    t0 = time()
    p = make_params(n_z=n_z, lambda=lambda, fixed_z=fixed_z)
    theta, E, v_pol = solve_labor_market(p)
    hh = solve_household(theta, E, v_pol, p)
    sim = simulate_population(theta, hh, p)
    elapsed = time() - t0
    tbl = entry_table(hh, p)
    n_enter = count(r -> r.min_a != "never", tbl)
    return (p=p, hh=hh, sim=sim, tbl=tbl, n_enter=n_enter, elapsed=elapsed)
end

grid_sizes = [15, 21]
results = Dict{Int,NamedTuple}()

println("="^64)
println("ABILITY GRID RESOLUTION CHECK  (lambda=1.0, persistent z, Pareto)")
println("="^64)

for nz in grid_sizes
    println("\n>>> n_z = $nz ...")
    r = run_nz(nz)
    results[nz] = r
    @printf("    runtime: %.1f min  |  entrepreneur share: %.2f%%  |  abilities entering: %d / %d\n",
        r.elapsed/60, 100*r.sim.share_e, r.n_enter, nz)
end

share15 = 100 * results[15].sim.share_e
share21 = 100 * results[21].sim.share_e
diff_pp = share21 - share15

io = open("/tmp/nz_resolution.txt", "w")
println(io, "ABILITY GRID RESOLUTION CHECK")
println(io, @sprintf("lambda=1.0  fixed_z=false  alpha_z=%.2f  z_min=%.2f", 1.35, 0.40))
println(io)
println(io, @sprintf("%-6s %12s %12s %10s %12s", "n_z", "ent_share%", "constr%", "runtime_min", "n_entering"))
println(io, "-"^56)
for nz in grid_sizes
    r = results[nz]
    @printf(io, "%-6d %12.2f %12.1f %10.1f %12d\n",
        nz, 100*r.sim.share_e, 100*r.sim.frac_constrained, r.elapsed/60, r.n_enter)
end
println(io)
@printf(io, "Share stability: n_z=21 minus n_z=15 = %+.2f pp\n", diff_pp)

for nz in grid_sizes
    r = results[nz]
    println(io, "\n--- Entry thresholds, n_z=$nz (entrepreneur share $(@sprintf("%.2f",100*r.sim.share_e))%) ---")
    println(io, @sprintf("  %-10s %-12s %-12s %-12s", "z", "min assets", "ss firm size", "k*(z,ss)"))
    println(io, "  " * "-"^48)
    for row in r.tbl
        cut_s = row.min_a isa String ? row.min_a : @sprintf("%.3f", row.min_a)
        marker = row.min_a == "never" ? "" : "  <-- enters"
        @printf(io, "  %-10.4f %-12s %-12.2f %-12.2f%s\n",
            row.z, cut_s, row.ell_ss, row.kstar, marker)
    end
end
close(io)

println("\n" * read("/tmp/nz_resolution.txt", String))
println("DONE_MARKER")
