using LinearAlgebra
using SparseArrays
using NonlinearEigenproblems 
using IterativeSolvers
using Plots

include("../src/NLA.jl")

#A0=[1 3 4; 5 6 2; 8 6 3]; A1=[3 4 8; 6 6 6; 0 7 1]; A2=[1 0 1; 0 1 0; 1 2 3];
#nep = PEP([A0,A1,A2]);

#nep=nep_gallery("dep0_tridiag");
nep=nep_gallery("qdep0");

max_iter = 200;
neigs = 20;
tol = 1e-7;

λ, v, err_hist, timings = NLA(nep, max_iter=max_iter, neigs=neigs, tol=tol);


# plots
err_hist[err_hist.==eps()] .= NaN;
p1=plot(1:max_iter, err_hist, yaxis=:log, lc=:black, lw=:2, legend=false)
p1=plot!(1:max_iter, ones(max_iter)*tol, ls=:dash, lc=:black, lw=:2, ylimits=(10.0^-10, 10.0^0+1), yticks=10.0 .^(-10:2:0), framestyle=:box)
title!("Errors from NLA")
ylabel!("Errs (log)")
xlabel!("Iters")

p2 = plot(1:max_iter-1, cumsum(timings[1:end-1,:], dims=1), lw=:1.5, label=["Total" "Expand projmats" "Inner solve" "Lin solve" "Orth"])
p2 = plot!(legend=:topleft)

savefig(p1, "../figs/NLA_qdep0.pdf")
savefig(p2, "../figs/NLA_timings.pdf")
