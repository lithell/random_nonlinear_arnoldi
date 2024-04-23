"""

Compute the shifted and scaled version of 'nep', with this custom shift and scale function for the 'gun'-benchmark. Although taylored with this in mind, it should technically work for any 'SPMFSumNEP' type, returning a 'SPMF_NEP'. Note that the 'SPMFSumNEP'-structure is lost, so use with care.
"""
function custom_shift_and_scale(
    nep::SPMFSumNEP;
    shift::Number=0,
    scale::Number=1
    )

    # get orig functions
    orgfv = get_fv(nep);
    m = size(orgfv,1);

    # create functions for the transformed problem
    fv = Vector{Function}(undef, m);

    for i = 1:m

        fv[i] = X -> orgfv[i](scale*X+shift*one(X))::((X isa Number) ? Number : Matrix)

    end

    # return transformed problem
    return SPMF_NEP(get_Av(nep), fv);

end

