"""
    SV, V, R = whiten_basis(SV::AbstractMatrix, V::AbstractMatrix)

Compute the thin QR decomposition of the sketched basis 'SV' and whiten basis by assigning 'SV' ← 'Q', 'V' ← 'V'('R'⁻¹). 
"""
function whiten_basis(SV::AbstractMatrix, V::AbstractMatrix)

    Q,R = qr(SV);
    Q = Matrix(Q);

    SV = Q;
    V = V/R;

    return SV, V;

end

function whiten_basis(
    SV::AbstractMatrix,
    V::AbstractMatrix,
    SAvV::AbstractArray
    )

    Q,R = qr(SV);
    Q = Matrix(Q);

    SV = Q;
    V = V/R;
    for i = 1:size(SAvV, 3)
        SAvV[:,1:end,i] = SAvV[:,1:end,i]/R;
    end
    return SV, V, SAvV;

end
