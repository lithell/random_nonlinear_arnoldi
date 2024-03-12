"""
    sketch_and_expand_projectmatrices!(nep::NEP, Vnew::AbstractMatrix, SVnew::AbstractMatrix, sketch::Function)

Sketching and expanding the projection matrices.
"""
function sketch_and_expand_projectmatrices!(
    nep::Proj_SPMF_NEP,
    Vnew::AbstractMatrix, 
    SVnew::AbstractMatrix,
    sketch::Function
    )

    # TODO: add option of computing the left projection via pseudo-inverse

    Sv = SVnew[:,end]; # latest sketched basis vec

    Av=nep.orgnep_Av; # original problem matrices
    k=size(Vnew, 2)-1; # current (before expansion) size of matrices
    m = size(nep.orgnep_Av,1); # number of terms 

    @assert(k+1 <= size(nep.projnep_B_mem[1],1)) # Don't go outside the prealloc memory

    # For over all i: Compute expanded part of matrices:
    
    B = map(i -> begin

            SAvV = sketch(Av[i]*Vnew); # can (maybe) do this better by saving SAvV iteratively during the algorithm. It's not very big

            # Expand the B-matrices
            nep.projnep_B_mem[i][1:k,(k+1)] = view(SVnew,:,1:k)'*SAvV[:,end]; 

            nep.projnep_B_mem[i][k+1,1:(k+1)] = (Sv)'*view(SAvV,:,1:(k+1));
            
            view(nep.projnep_B_mem[i],1:(k+1),1:(k+1));

            end, 1:m)

    # Keep the sequence of functions for SPMFs
    nep.nep_proj=SPMF_NEP(B,nep.orgnep_fv,check_consistency=false)

end



function sketch_and_expand_projectmatrices!(
    nep::Proj_SPMF_NEP,
    Vnew::AbstractMatrix, 
    SVnew::AbstractMatrix,
    SAvVnew::AbstractArray,
    sketch::Function
    )

    # TODO: add option of computing the left projection via pseudo-inverse

    Sv = SVnew[:,end]; # latest sketched basis vec

    Av=nep.orgnep_Av; # original problem matrices
    k=size(Vnew, 2)-1; # current (before expansion) size of matrices
    m = size(nep.orgnep_Av,1); # number of terms 

    @assert(k+1 <= size(nep.projnep_B_mem[1],1)) # Don't go outside the prealloc memory

    # For over all i: Compute expanded part of matrices:
    
    B = map(i -> begin

            # Expand the B-matrices
            nep.projnep_B_mem[i][1:k,(k+1)] = view(SVnew,:,1:k)'*SAvVnew[:,end,i]; 

            nep.projnep_B_mem[i][k+1,1:(k+1)] = (Sv)'*view(SAvVnew,:,1:(k+1),i);
            
            view(nep.projnep_B_mem[i],1:(k+1),1:(k+1));

            end, 1:m)

    # Keep the sequence of functions for SPMFs
    nep.nep_proj=SPMF_NEP(B,nep.orgnep_fv,check_consistency=false)

end
