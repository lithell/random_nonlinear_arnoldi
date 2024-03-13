using LinearAlgebra
using NonlinearEigenproblems
using SparseArrays
using Plots
using Random

include("../src/gen_perturbed_example.jl")
include("../src/gen_sparse_mat.jl")

Random.seed!(1);

n = 500;
extreme_radius = 5;
σ = 2;

Σ = [ 5 .+ rand(n-1).*exp.(2im*pi*rand(n-1)); 5 + extreme_radius]

Σ = σ .+ 1 ./Σ;


A = gen_sparse_mat(Σ);

ll = eigvals(Matrix(A));

dd = scatter(1 ./(Σ .- σ))
dd = scatter!(1 ./(ll .- σ), markershape=:cross, markersize=10, aspect_ratio=:equal)
display(dd)

