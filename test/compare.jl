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

Random.seed!(0);

n = 500;
σ = 3;
extreme_radius = 8;
linear=false;

nep, λ_ref, v_ref, eig_cond = gen_perturbed_example(
    n,
    σ,
    extreme_radius,
    linear=linear
    );

# set params
max_iter = 50;
neigs = 1;
tol = 1e-18;
s = 2*max_iter;
trunc_len = 4;
save_reduced_matrices=true;

sketch = setup_sketching_handle(n, s);

z = randn(ComplexF64, n);
normalize!(z)
vstart=deepcopy(z); # vstart is modified in the functions

λ = σ;

inner_solver_method = NEPSolver.IARInnerSolver();
#inner_solver_method = NEPSolver.NewtonInnerSolver();
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
    markershape=:utriangle,
    markercolor=:white,
    markersize=4,
    lw=:1.2,
    label="NLAR"
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
    markershape=:circle,
    markercolor=:white,
    markersize=4,
    lw=:1.2,
    label="sNLAR"
    )

# plot tol
most_iters = max(length(err_hist_NLA), length(err_hist_sNLA));
#p1 = plot!(
#    1:most_iters, 
#    tol*ones(most_iters),
#    yaxis=:log,
#    lc=:black,
#    ls=:dot,
#    lw=:1.2,
#    label="tol"
#    )


# plot eigval cond number times eps_mach
p1 = plot!(
    1:most_iters, 
    eps()*eig_cond*ones(most_iters),
    yaxis=:log,
    lc=:black,
    ls=:dashdotdot,
    lw=:1.2,
    label="κ(λ_ref)⋅ε_mach"
    )


# more plot styling
p1 = plot!(
    framestyle=:box,
    size=(900,500),
    minorticks=true,
    ylimits=(10.0^-12, 10.0^0),
    yticks=10.0 .^(-12:2:0),
    left_margin = 10mm,
    right_margin = 10mm,
    grid=false
    )

ylabel!(L"|\lambda_{ref} - \widehat{\lambda}\:|")
xlabel!(L"\mathrm{Iterations}")
#title!("Convergence of NLA & sNLA (n=$n, s=$s, trunc_len=$trunc_len)")

# plot sparsity of linear factor
B = copy(abs.(get_Av(nep)[1]));
B[findall(x -> abs(x) > 0, B)] .= 1; # ugly but just want for plot

p2 = spy(
    B,
    aspect_ratio=:equal,
    legend=nothing,
    framestyle=:box,
    size=(500,500),
    markersize = 0.9
    )


display(p1)
