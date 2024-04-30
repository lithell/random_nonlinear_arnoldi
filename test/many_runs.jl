using LinearAlgebra
using SparseArrays
using FFTW
using NonlinearEigenproblems 
using Random
using IterativeSolvers
using Plots
using Printf
using Measures
using LaTeXStrings

include("../src/NLA.jl")
include("../src/sNLA.jl")
include("../src/setup_sketching_handle.jl")
include("../src/whiten_basis.jl")
include("../src/sketch_and_expand_projectmatrices.jl")
include("../src/gen_perturbed_example.jl")
include("../src/sketch_reduced_matrices.jl")
include("../src/custom_shift_and_scale.jl")


# params
max_iter = 20;
s = 4*max_iter;
neigs = 1;
tol=1e-18;
shift = 250^2;
scale = 300^2 - 200^2;
λ=0.5;

trunc_len = 4;
save_reduced_matrices = true;

# set up problem 
nep = nep_gallery("nlevp_native_gun");
nep1 = custom_shift_and_scale(nep, scale=scale, shift=shift);
n = size(nep1, 1);

num_runs = 20;

errs = zeros(max_iter, num_runs);

for i = 1:num_runs

    Random.seed!(i);

    local sketch = setup_sketching_handle(n, s);

    # solve problem by sNLAR
    local (λ_snlar, v_snlar, err_hist_snlar) = sNLA(
        nep1, 
        sketch, 
        s,
        trunc_len=trunc_len,
        max_iter=max_iter, 
        neigs=neigs, 
        tol=tol,
        λ=λ,
        inner_solver_method=IARInnerSolver()
        );

    errs[:,i] = err_hist_snlar;

end


# plot convergence
errs[errs.==eps()] .= NaN;
#errs = errs[:];
for i = 1:num_runs
    deleteat!(errs[:,i], findall(isnan, errs[i]))
end

worst_run = maximum(errs, dims=2);
best_run = minimum(errs, dims=2);

p1 = plot(1:max_iter, best_run, fillrange=worst_run, 
    alpha=0.5,
    fillstyle=:/,
    fillcolor=:gray,
    label="Best-worst interval",
    linealpha=1,
    lw=:1.2,
    linestyle=:dash,
    lc=:black,
    framstyle=:box
    );

p1 = plot!(1:max_iter, worst_run, fillrange=best_run, 
    alpha=0,
    fillcolor=:gray,
    linealpha=1,
    lw=:1.2,
    linestyle=:dash,
    lc=:black,
    framstyle=:box,
    label=:none
    );

p1 = plot!(
    1:max_iter,
    errs,
    yaxis=:log,
    lc=:black,
    lw=:0.6,
    label=:none
    )

p1 = plot!(
    framestyle=:box,
    size=(900,500),
    minorticks=true,
    ylimits=(0.5*10.0^-18, 10.0^0),
    yticks=10.0 .^(-18:2:0),
    xlimits=(0, max_iter),
    xticks=0:5:max_iter,
    left_margin = 10mm,
    right_margin = 10mm,
    grid=true,
    legend=:topright
    )

xlabel!(L"\mathrm{Iterations}")
ylabel!(L"\mathrm{Residual\:\:norm}")




