"""
    sketching_handle = setup_sketchin_handle(N:Int, sketch_param::Int)

Set up a sketching function 'sketching_handle' for size 'N'(×'N') input with embedding dimension 'sketch_param'. The sketching function returned performs a SRCT on the input.
"""
function setup_sketching_handle(N::Int, sketch_param::Int)

    if sketch_param > N
        @error "Notice! 'sketch_param' larger than problem size. Exiting with value Nothing." 
        return 
    end

    # construct random E
    rows = 1:N;
    cols = 1:N;
    vals = 2*round.(rand(N)).-1;
    E = sparse(rows, cols, vals);

    # construct random D
    D = sparse(I, N, N);
    D = D[rand(1:N, sketch_param), :];

    # Return sketching function
    return x -> D*dct(E*x,1)/sqrt(sketch_param/N);

end
