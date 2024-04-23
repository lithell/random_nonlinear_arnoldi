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
include("../src/custom_shift_and_scale.jl")

Random.seed!(123);

# params
max_iter = 30;
s = 6*max_iter;
neigs = 20;
tol=1e-8;
shift = 250^2;
scale = 300^2 - 200^2;
λ=0.5;

trunc_len = 4;
save_reduced_matrices = true;

# set up problem 
nep = nep_gallery("nlevp_native_gun");
nep1 = custom_shift_and_scale(nep, scale=scale, shift=shift);
n = size(nep1, 1);

sketch = setup_sketching_handle(n, s);

z = randn(ComplexF64, n);
normalize!(z)
vstart=deepcopy(z); # vstart is modified in the functions

# solve problem by NLAR
λ_nlar, v_nlar, err_hist_nlar = NLA(
    nep1,
    max_iter=max_iter,
    neigs=neigs,
    tol=tol,
    λ=λ,
    inner_solver_method=IARInnerSolver()
    );

# solve problem by sNLAR
λ_snlar, v_snlar, err_hist_snlar = sNLA(
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


# plot nlar results
p1 = scatter(
    sqrt.(λ_nlar.*scale .+ shift),
    lc=:black,
    markershape=:square,
    markercolor=:white,
    markersize=4.5,
    ylimits=[-10, 150],
    xlimits=[80, 340],
    xticks=(80:20:340),
    yticks=(0:50:150),
    label="NLAR"
    )

# plot snlar results
p1 = scatter!(
    sqrt.(λ_snlar.*scale .+ shift),
    lc=:black,
    markershape=:circle,
    markercolor=:white,
    markersize=4,
    ylimits=[-5, 150],
    xlimits=[80, 340],
    xticks=(80:20:340),
    yticks=(0:50:150),
    label="sNLAR"
    )

p1 = plot!(legend=:topleft)
ylabel!(L"\mathrm{Im}\:\:\sqrt{λ}")
xlabel!(L"\mathrm{Re}\:\:\sqrt{λ}")
