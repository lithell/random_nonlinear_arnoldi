import NonlinearEigenproblems.default_eigval_sorter # for fixing wierd behaviour of function

"""
    λ, v, err_hist, timings = NLA(nep::ProjectableNEP, kwargs)

Compute the Nonlinear Arnoldi (NLA) approximations of the eigenvalues of 'nep'. 

If 'kwargs' left unspecified default values are used.

This code is heavily inspired by, and lends many parts of, the implementation of NLA in NEP-PACK. 

See: https://nep-pack.github.io/NonlinearEigenproblems.jl/dev/
"""
NLA(nep::NEP;params...) = NLA(ComplexF64,nep::NEP;params...)
function NLA(
    ::Type{T},
    nep::ProjectableNEP; # optional args from here on 
    max_iter::Int=100, 
    neigs::Int=10,
    tol = eps()*1000,
    R = 0.01,
    λ::Number = 0,
    v::Vector = randn(ComplexF64, size(nep, 1)),
    errmeasure::ErrmeasureType = DefaultErrmeasure(nep),
    linsolvercreator=DefaultLinSolverCreator(),
    eigval_sorter::Function = default_eigval_sorter,
    orthmethod = ModifiedGramSchmidt(), # think this is not used
    inner_solver_method = DefaultInnerSolver()
    ) where {T<:Number};

    local linsolver::LinSolver=create_linsolver(linsolvercreator,nep,λ)

    n = size(nep, 1);
    
    # check that max_iter is not larger than problem size 
    if (max_iter > n)
        @error "Notice! 'max_iter' larger than problem size. Exiting with value Nothing.\n"
        return nothing;
    end

    # init stuff
    V = zeros(T, n, max_iter); # store basis
    D = zeros(T, neigs); # store conv eigs
    X = zeros(T, n, neigs); # store conv eigvecs
    timings = zeros(max_iter, 5); # store timings of different parts
    err_hist=eps()*ones(max_iter,neigs); # err history 
    σ = λ; # intital pole (not sure where this is updated, but seems to work?)
    k = 1; # iters
    m = 0; # num conv eigs
    u::Vector{T} = v;
    ν::T = λ;

    # add first basis vector
    normalize!(v)
    V[:,1] = v;
    
    # create projected NEP
    pnep = create_proj_NEP(nep, max_iter, T);

    while ((m < neigs) && (k < max_iter))
        
        # timing
        tic_tot = Int(time_ns());

        Vk = view(V, :, 1:k);

        # expand projection matrices
        tic_proj = Int(time_ns());

        expand_projectmatrices!(pnep, Vk, Vk);
        
        toc_proj = Int(time_ns());
        time_proj = 1e-9*(toc_proj-tic_proj);
        timings[k,2] = time_proj;

        # solve projected problem 
        tic_inner = Int(time_ns());

        dd, vv = inner_solve(inner_solver_method,T,pnep,neigs=neigs,σ=σ,tol=1e-16, maxit=100); # TODO: fix this hardcode (maybe)

        toc_inner = Int(time_ns());
        time_inner = 1e-9*(toc_inner-tic_inner);
        timings[k,3] = time_inner;

        # sort eigs of projected problem 
        νv, yv = eigval_sorter(nep, dd, vv, σ, D, R, Vk);

        ν = νv[1]; y = yv[:,1];

        u[:] = Vk*y;

        # normalize and compute res
        normalize!(u)
        res = compute_Mlincomb(nep, ν, u);

        # check for converged eigs
        err = estimate_error(errmeasure, ν, u);

        # log error 
        err_hist[k,m+1] = err; 

        # add any converged eigs
        if (err < tol)

            D[m+1] = ν;
            X[:,m+1] = u;

            # sort and select eigs again (not sure why we do this step?)
            νv,yv = eigval_sorter(nep, dd, vv, σ, D, R, Vk);
            ν1=νv[1];
            y1=yv[:,1];

            # compute res again
            u1 = Vk*y1;
            normalize!(u1);
            res = compute_Mlincomb(nep, ν1, u1);

            m = m+1;

        end

        # maybe put a restart here?

        # compute next search direction
        tic_linsol = Int(time_ns());

        Δv = lin_solve(linsolver, res);

        toc_linsol = Int(time_ns());
        time_linsol = 1e-9*(toc_linsol-tic_linsol);
        timings[k,4] = time_linsol;

        # orthogonalize (DGS)
        tic_ort = Int(time_ns());

        for i = 1:2
            Δv = Δv - Vk*Vk'*Δv;
        end

        toc_ort = Int(time_ns());
        time_ort = 1e-9*(toc_ort-tic_ort);
        timings[k, 5] = time_ort;

        normalize!(Δv)

        # expand basis
        V[:,k+1] = Δv;


        # timing
        toc_tot = Int(time_ns());
        time_tot = 1e-9*(toc_tot-tic_tot);
        timings[k,1] = time_tot;

        k += 1

    end

    if ((k > max_iter) || (m < neigs))
        @warn "No convergence to within tol achieved with NLA. Found $m eigs out of $neigs wanted. Latest eigenvalue iterate: \n" ν
    end

    return D[1:m], X[:,1:m], err_hist, timings;
    
end


function discard_ritz_values!(dd,D,R)
    for i=1:size(dd,1)
        for j=1:size(D,1)
            if (abs(dd[i]-D[j])<R)
                dd[i]=Inf; #Discard all Ritz values within a particular radius R
            end
        end
    end

end


function  default_eigval_sorter(nep::NEP,dd,vv,σ,D,R,Vk)
    dd2=copy(dd);

    #Discard ritz values within a distance R of the converged eigenvalues
    discard_ritz_values!(dd2,D,R)

    ii = sortperm(abs.(dd2.-σ));

    nu = dd2[ii];
    y = vv[:,ii];

    return nu,y;
end
