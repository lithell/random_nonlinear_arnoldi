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

Random.seed!(123);

# params
max_iter = 40;
s = 4*max_iter;
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
    v=vstart,
    inner_solver_method=IARInnerSolver()
    );

vstart=deepcopy(z);
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
    v=vstart,
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

# Liao et. al.
c=3e8;

Qstar1=[34643.66;0.36;0.48;2136.73;12376.84;1149.21;7714.93;118.71;3.17;3.59];
fj1=[7.1373;4.5655;7.2465;9.9992;10.0449;10.4762;10.5463;11.1581;14.2617;15.0143];
resqrt1=2*pi*fj1*1e9/c;

imsqrt1=0.5*resqrt1./Qstar1;

Qstar2=[34643.66;2136.73;12376.84;1149.21;7714.93;118.71;15.25;536.76;2500.75;181.23]
fj2=[7.1373;9.9992;10.0449;10.4762;10.5463;11.1518;13.1180;13.2698;13.5882;13.7688];
resqrt2=2*pi*fj2*1e9/c;
imsqrt2=0.5*resqrt2./Qstar2;

liaoevps=[resqrt1;resqrt2]+1im*[imsqrt1;imsqrt2];

p1 = scatter!(
    liaoevps, 
    lc=:black,
    markershape=:cross,
    markercolor=:black,
    markersize=4,
    ylimits=[-5, 150],
    xlimits=[80, 340],
    xticks=(80:20:340),
    yticks=(0:50:150),
    label="Liao et al."
    ) 

p1 = plot!(legend=:topleft, framestyle=:box)
ylabel!(L"\mathrm{Im}\:\:\sqrt{λ}")
xlabel!(L"\mathrm{Re}\:\:\sqrt{λ}")


# plot convergence
err_hist_nlar[err_hist_nlar.==eps()] .= NaN;
err_hist_nlar = err_hist_nlar[:];
deleteat!(err_hist_nlar, findall(isnan, err_hist_nlar));

p2 = plot(
    1:length(err_hist_nlar), 
    err_hist_nlar,
    yaxis=:log,
    lc=:black,
    ls=:dash,
    #markershape=:utriangle,
    #markercolor=:white,
    markersize=4,
    lw=:1.2,
    label="NLAR"
    )


err_hist_snlar[err_hist_snlar.==eps()] .= NaN;
err_hist_snlar = err_hist_snlar[:];
deleteat!(err_hist_snlar, findall(isnan, err_hist_snlar));

p2 = plot!(
    1:length(err_hist_snlar), 
    err_hist_snlar,
    yaxis=:log,
    lc=:black,
    #markershape=:circle,
    #markercolor=:white,
    markersize=4,
    lw=:1.2,
    label="sNLAR"
    )

# plot styling
p2 = plot!(
    framestyle=:box,
    size=(900,500),
    minorticks=true,
    ylimits=(0.5*10.0^-10, 10.0^0),
    yticks=10.0 .^(-10:2:0),
    xlimits=(0, max_iter),
    xticks=0:5:max_iter,
    left_margin = 10mm,
    right_margin = 10mm,
    grid=true
    )

ylabel!(L"\mathrm{Residual\:\: norm}")
xlabel!(L"\mathrm{Iterations}")
