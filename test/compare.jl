using LinearAlgebra
using SparseArrays
using FFTW
using NonlinearEigenproblems 
using Random
using IterativeSolvers
using Plots
using Printf
using Measures

include("../src/NLA.jl")
include("../src/sNLA.jl")
include("../src/setup_sketching_handle.jl")
include("../src/whiten_basis.jl")
include("../src/sketch_and_expand_projectmatrices.jl")
include("../src/gen_perturbed_example.jl")
include("../src/sketch_reduced_matrices.jl")

Random.seed!(321);

n = 2000;
σ = 3;
extreme_radius = 5;
linear=false;

nep, ref_eig = gen_perturbed_example(
    n,
    σ,
    extreme_radius,
    linear=linear
    );

# set params
max_iter = 30;
neigs = 1;
tol = 1e-10;
s = 4*max_iter;
trunc_len = 6;
save_reduced_matrices=true;

sketch = setup_sketching_handle(n, s);

z = randn(ComplexF64, n);
normalize!(z)
vstart=deepcopy(z); # vstart is modified in the functions

λ = σ;

λ_ref = ref_eig;

inner_solver_method = NEPSolver.IARInnerSolver();
errmeasure = EigvalReferenceErrmeasure(nep, λ_ref);

# solve problem by NLA
λ_NLA, v_NLA, err_hist_NLA = NLA(
    nep,
    max_iter=max_iter,
    neigs=neigs,
    tol=tol,
    v=vstart,
    λ=λ,
    inner_solver_method=inner_solver_method,
    errmeasure=errmeasure
    );


# plot convergence
err_hist_NLA[err_hist_NLA.==eps()] .= NaN;
err_hist_NLA = err_hist_NLA[:];
deleteat!(err_hist_NLA, findall(isnan, err_hist_NLA));

p1 = plot(
    1:length(err_hist_NLA), 
    err_hist_NLA,
    yaxis=:log,
    lc=:black,
    lw=:1.2,
    label="NLA"
    )


# solve problem by sNLA
vstart=deepcopy(z);

λ_sNLA, v_sNLA, err_hist_sNLA = sNLA(
    nep,
    sketch,
    s,
    max_iter=max_iter,
    trunc_len=trunc_len,
    neigs=neigs,
    tol=tol,
    v=vstart,
    λ=λ,
    inner_solver_method=inner_solver_method,
    errmeasure=errmeasure,
    save_reduced_matrices=save_reduced_matrices
    );


# plot convergence
err_hist_sNLA[err_hist_sNLA.==eps()] .= NaN;
err_hist_sNLA = err_hist_sNLA[:];
deleteat!(err_hist_sNLA, findall(isnan, err_hist_sNLA));

p1 = plot!(
    1:length(err_hist_sNLA), 
    err_hist_sNLA,
    yaxis=:log,
    lc=:black,
    ls=:dash,
    lw=:1.2,
    label="sNLA"
    )

# plot tol
most_iters = max(length(err_hist_NLA), length(err_hist_sNLA));
p1 = plot!(
    1:most_iters, 
    tol*ones(most_iters),
    yaxis=:log,
    lc=:black,
    ls=:dot,
    lw=:2,
    label="tol"
    )

p1 = plot!(
    framestyle=:box,
    size=(900,500),
    minorticks=true,
    ylimits=(10.0^-18, 10.0^2),
    yticks=10.0 .^(-18:2:2),
    left_margin = 10mm,
    right_margin = 10mm
    )

ylabel!("Absolute Error in Eigenvalue")
xlabel!("Iterations")
title!("Convergence of NLA & sNLA (n=$n, s=$s, trunc_len=$trunc_len)")

# plot sparsity of linear factor
p2 = spy(
    abs.(get_Av(nep)[1]),
    aspect_ratio=:equal,
    legend=:none,
    framestyle=:box,
    size=(500,500)
    )

title!("Sparsity Pattern of Linear Factor")

display(p1)
