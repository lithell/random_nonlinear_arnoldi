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
max_iter = 20;
s = 2*max_iter;
neigs = 1;
tol=1e-7;
shift = 250^2;
scale = 300^2 - 200^2;

truc_len = 4;
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
    λ=1,
    inner_solver_method=IARInnerSolver()
    );






