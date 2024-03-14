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
    z = [ 5 .+ rand(n-1).*exp.(2im*pi*rand(n-1)); 5 + extreme_radius]

    Σ = σ .+ 1 ./z;

    A = gen_sparse_mat(Σ);
    f1 = λ -> one(λ);

    B = sparse(Matrix(-1*I, n, n));
    f2 = λ -> λ;

    nep1 = SPMF_NEP([A, B], [f1, f2]);

    # low rank
    c = randn(n,1)/1e2;
    d = randn(n,1)/1e2;
    f3 = λ -> 1 ./ (5 .- λ); # fix this hardcode. Probably should be careful with this...

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
        v_ref = vv[:,kk];
    else 

        # compute some eigs and hope it includes the one we want... probably think of a better way than this, but should work for our contrived examples
        λ, v_ref = try 
            iar(nep; neigs=5, tol=5e-5, maxit=120, σ=σ) 
        catch err
            err.λ, err.v;
        end

        λ_ref = λ[findmax(abs.( 1 ./(λ .- σ)))[2]];
        v_ref = v_ref[:,1]

        # run a few Newton iters to be sure we found the eig to good precision
        λ_ref, v_ref = augnewton(nep, maxit=30, λ=λ_ref, v=v_ref, tol=1e-16);

    end

    # compute eigenvalue condition number 
    if linear 
        λ_ref_left, w_ref = eigen(Matrix(A)');
        j = argmin(abs.(λ_ref_left .- λ_ref));
        w_ref = w_ref[:,j];
        eig_cond = 1/abs.(w_ref'*v_ref);
    else 
        M = compute_Mder(nep, λ_ref, 0);
        λ_ref_left, w_ref = eigen(Matrix(M)');
        j = argmin(abs.(λ_ref_left .- λ_ref));
        w_ref = w_ref[:,j];
        eig_cond = 1/abs(w_ref'*compute_Mder(nep, λ_ref, 1)*v_ref);
    end

    @info "Constructed problem with reference eigenvalue & eigenvalue condition number:" λ_ref eig_cond

    return nep, λ_ref, v_ref;
end

