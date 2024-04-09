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

Random.seed!(123);

# params
#n = 1000;
max_iter = 50;
s = 2*max_iter;
neigs = 10;
tol=1e-7;
truc_len = 4;
save_reduced_matrices = true;
#inner_solver_method = NewtonInnerSolver();


# set up problem 
nep = nep_gallery("qdep0");

sketch = setup_sketching_handle(n, s);

z = randn(ComplexF64, n);
normalize!(z)
vstart=deepcopy(z); # vstart is modified in the functions

# solve problem by NLAR
λ_nlar, v_nlar, err_hist_nlar = NLA(
    nep,
    max_iter=max_iter,
    neigs=neigs,
    tol=tol,
    inner_solver_method=IARInnerSolver(),
    );






