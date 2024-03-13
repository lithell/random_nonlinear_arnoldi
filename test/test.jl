using LinearAlgebra
using NonlinearEigenproblems
using SparseArrays
using Plots
using Random

include("../src/gen_perturbed_example.jl")
include("../src/gen_sparse_mat.jl")

#Random.seed!(1);
#
##Σ = rand(100);
#
#n = 1000;
#Σ = [randn(n-1).*exp.(2im*pi*rand(n-1)); 5]
##center = 10;
##radius = 2;
##extreme_dist = 5;
##k = 2π/(n-1);
##tv = k*(1:(n-1));
##Σ = [center .+ radius*rand(n-1) .*exp.(tv*1im); center + extreme_dist];
#
#A = gen_sparse_mat(Σ);
#
#
##h = heatmap(log.(abs.(A)), aspect_ratio=:equal);
##h = heatmap((abs.(A)), aspect_ratio=:equal);
##display(h)
#
#scatter(eigvals(Matrix(A)))
#scatter!(Σ, markershape=:cross, aspect_ratio=:equal)

x = try 
    sqrt(-3)
catch
    @warn "!!"
    3
end
