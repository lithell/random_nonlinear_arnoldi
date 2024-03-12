function sketch_reduced_matrices!(
    SAvV::AbstractArray,
    nep::Proj_SPMF_NEP,
    v::Vector,
    k::Int
    )

    Av=nep.orgnep_Av; # original problem matrices
    num_terms = size(Av,1);

    for i = 1:num_terms
        SAvV[:,k, i] =  sketch(Av[i]*v);
    end

end
