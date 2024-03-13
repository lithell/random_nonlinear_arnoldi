"""
Σ is vec of eigs wanted
"""
function gen_sparse_mat(Σ)

    n = length(Σ);

    II=sortperm(randn(n))
    wanted_evps=Σ[II];

    T=spdiagm(0 => log.(wanted_evps),
              -1 => 0.003*randn(n-1), 1 => 0.003*randn(n-1),
              -5 => 0.005*randn(n-5));

    A=exp(Matrix(T))

    A[findall(x -> abs(x) < 1e-10, A)] .= 0;

    A=sparse(A)

    A=A+sprandn(n,n,0.01)*0.01 # Not just a band-matrix


    numnz = nnz(A);
    numels = n^2;
    perc = (1-numnz/numels)*100;

    @info "Generated sparse matrix with $numnz nonzero elements ($perc % sparsity)" 

    return A;

end





