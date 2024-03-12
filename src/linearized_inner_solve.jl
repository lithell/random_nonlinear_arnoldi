"""
    λ, v = linearized_inner_solve(nep::NEP, Vlnew::AbstractMatrix, Vrnew::AbstractMatrix, kwargs)

Compute the eigs of the projected problem 'nep' by Rosenbrock linearization. The function accepts the additional flag 'sketched' which if set to 'true' will use the QR-decomposed form for applying the sketched matrices. Notice that the matrix 'R' as well as the sketched reduced problem matrices must be supplied in this case.

Notice! This function is taylored to an example specific to this work, so it should not be expected to work well, if at all, for other problems. However, if a low-rank nep is defined as in this example, it should work.

IN PROGRESS
"""
linearized_inner_solve(nep::NEP, Vlnew::AbstractMatrix, Vrnew::AbstractMatrix; params...) = linearized_inner_solve(ComplexF64,nep::NEP, Vlnew::AbstractMatrix, Vrnew::AbstractMatrix; params...)
function linearized_inner_solve(
    ::Type{T},
    nep::ProjectableNEP,
    Vlnew::AbstractMatrix, 
    Vrnew::AbstractMatrix,
    nonlin_shift::Number;
    sketched::Bool = false,
    R::AbstractMatrix
    )

    k = size(Vlnew)[2];

    if sketched == false

        # construct linearized system 
        A = [
            nep.nep2.d'*Vrnew  nonlin_shift;
            Vlnew'*get_Av(nep.nep1)[1]*Vrnew  Vlnew'*(nep.nep2.c)
            ];

        B = [
            zeros(k)' -1;
            -Vlnew'*get_Av(nep.nep1)[2]*Vrnew  zeros(k)
            ];

        # solve by QR-type method
        F = eigen(A,B);

        λ, v = F;

        return λ, v;

    else

    

    end


    


end
    
