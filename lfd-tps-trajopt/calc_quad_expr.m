function [Q, q, c] = calc_quad_expr(f, shape)
    %CALC_QUAD_EXPR converts f to the matrix form.
    %
    % [Q, q, c] = calc_quad_expr(f, [n,1])
    %
    % f must be a quadratic function
    %
    % Note that f must not create conjugates, i.e. for transposing A, do 
    % explicit transpose A.' instead of conjugate transpose A'
    % 
    % Note c is not correct.
    %
    % This function finds Q, q and c such that
    % (1/2)*(x.'*Q*x) + q*x + c = f(x)
    
    if shape(1)~=1 && shape(2)~=1
        error('shape of variable should be a vector');
    end
    if shape(1)~=1
        n = shape(1);
    else
        n = shape(2);
    end
    
    x_sym = sym('x', [n 1]);
    quad_obj = f(x_sym);
    Q = jacobian(jacobian(quad_obj, x_sym), x_sym);
    lin_obj = quad_obj - (1/2)*(x_sym.'*Q*x_sym);
    q = jacobian(lin_obj, x_sym);
    const = lin_obj - q*x_sym;
    
    if jacobian(const, x_sym) ~= 0 % const is not a constant
        error('given function cannot be expressed in quadratic form');
    end
    if nargout > 2
        c = simplify(const);
        try
            c = double(c);
        catch
            error('given function cannot be expressed in quadratic form');
        end
    end
 
    try
        Q = double(Q);
    catch
        try
            Q = double(simplify(Q));
        catch
            error('given function cannot be expressed in quadratic form');
        end
    end
    
    try
        q = double(q);
    catch
        try
            q = double(simplify(q));
        catch
            error('given function cannot be expressed in quadratic form');
        end
    end
end
