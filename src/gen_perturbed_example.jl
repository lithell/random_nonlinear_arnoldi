include("../src/gen_sparse_mat.jl")
"""
    nep = gen_perturbed_example(n::Int, σ::Number, extreme_radius::Number, kwargs)

Generate NEP that is a perturbed linear eigenproblem with a low rank non-linear term. 'nep' is of size 'n'. 'σ' is chosen to be close to the inital shift in the solver method (shift-and-invert-type solver). 'extreme_radius' is distance from center of eig-disc to outlier.   

The function accepts the additional parameter 'linear', which when set to true will return the linear problem, i.e. without the perturbation. It can also optionally return the variable 'λ_ref' which is (hopefully) the true (outlier) eig of the problem. This might not be robust so proceed with caution.

Notice that this function generates an intermediate full matrix, and so might not be suitable for generating very large problems.
"""
function gen_perturbed_example(
    n::Int,
    σ::Number,
    extreme_radius::Number;
    linear::Bool = false
    ) 

    # linear problem with "known" eigs
    z = [randn(n-1).*exp.(2im*pi*rand(n-1)); extreme_radius]

    Σ = σ .+ 1 ./z;

    A = gen_sparse_mat(Σ);
    f1 = λ -> one(λ);

    B = sparse(Matrix(-1*I, n, n));
    f2 = λ -> λ;

    nep1 = SPMF_NEP([A, B], [f1, f2]);

    # low rank
    c = randn(n,1)/1e2;
    d = randn(n,1)/1e2;
    f3 = λ -> 1 ./ (5 .- λ); # fix this hardcode. Probably should be carefull with this...

    nep2 = LowRankFactorizedNEP([c],[d],[f3]);

    
    if linear
        nep = nep1;
    else 
        nep = SumNEP(nep1, nep2);
    end

    # compute reference eigenvalue
    if linear
        ll, vv = eigen(Matrix(A));
        kk = findmax(abs.( 1 ./(ll .- σ)))[2]; # this might not be robust...
        λ_ref = ll[kk];
        v_ref = vv;
    else 

        # compute one eigs and hope it's the one we want... probably think of a better way than this
        λ, v_ref = try 
            iar(nep; neigs=1, tol=5e-12, maxit=120, σ=σ) 
        catch err
            err.λ, err.v;
        end


        λ_ref = λ[findmax(abs.( 1 ./(λ .- σ)))[2]];
        v_ref = v_ref[:,1]

        # run a few Newton iters to be sure we found the eig to good precision
        λ_ref, v_ref = augnewton(nep, maxit=10, λ=λ_ref, v=v_ref);

    end

    @info "Constructed problem with reference eigenvalue:" λ_ref

    return nep, λ_ref, v_ref;
end

