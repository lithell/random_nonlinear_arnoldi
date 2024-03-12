using LinearAlgebra
using SparseArrays
using FFTW
using NonlinearEigenproblems 
using Random
using IterativeSolvers
using Plots

include("../src/sNLA.jl")
include("../src/setup_sketching_handle.jl")
include("../src/whiten_basis.jl")
include("../src/sketch_and_expand_projectmatrices.jl")

Random.seed!(1321);

#nep=nep_gallery("dep0_tridiag");
nep=nep_gallery("qdep0");

n = size(nep, 1);

max_iter = 200;
neigs = 20;
tol = 1e-7;
s = 2*max_iter;
trunc_len = 4;

sketch = setup_sketching_handle(n, s);

λ, v, err_hist, timings = sNLA(nep, sketch, s, max_iter=max_iter, trunc_len = trunc_len, neigs=neigs, tol=tol)

# plots
err_hist[err_hist.==eps()] .= NaN;
p1=plot(1:max_iter, err_hist, yaxis=:log, lc=:black, lw=:2, legend=false)
p1=plot!(1:max_iter, ones(max_iter)*tol, ls=:dash, lc=:black, lw=:2, ylimits=(10.0^-10, 10.0^0+1), yticks=10.0 .^(-10:2:0), framestyle=:box)
title!("Errors from sNLA")
ylabel!("Errs (log)")
xlabel!("Iters")


p2 = plot(1:max_iter-1, cumsum(timings[1:end-1,:], dims=1), lw=:1.5, label=["Total" "Basis whitening" "Expand projmats" "Inner solve" "Lin solve" "Orth"])
p2 = plot!(legend=:topleft)

#savefig(p, "../figs/dep0_tridiag.pdf")
savefig(p1, "../figs/sNLA_qdep0.pdf")
savefig(p2, "../figs/sNLA_timings.pdf")
