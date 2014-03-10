function [A, c] = calc_lin_expr(fs, shape)
    %CALC_LIN_EXPR converts fs to the matrix form.
    %
    % [A, c] = calc_lin_expr(fs, [n,1])
    %
    % fs must be a single linear function or a cell of linear functions
    %
    % Note that the functions must not create conjugates, i.e. for 
    % transposing A, do explicit transpose A.' instead of conjugate 
    % transpose A'
    % 
    % If fs is a single function, this function finds A and c such that
    % A*x + c = reshape(fs(x), [] ,1)
    %
    % If fs is a cell of functions, this function finds A and c such that
    % A*x + c = [ reshape(fs{1}(x), [] ,1);
    %                        ...
    %             reshape(fs{i}(x), [] ,1);
    %                        ...
    %             reshape(fs{n}(x), [] ,1) ]
    
    if ~iscell(fs)
        [A, c] = calc_single_lin_exp(fs, shape);
    else
        if shape(1)~=1 && shape(2)~=1
            error('shape of variable should be a vector');
        end
        if shape(1)~=1
            n = shape(1);
        else
            n = shape(2);
        end
        
        n_functions = length(fs);
        A_cell = cell(n_functions,1);
        c_cell = cell(n_functions,1);
        m = 0;
        for i=1:n_functions
            [A_cell{i}, c_cell{i}] = calc_single_lin_exp(fs{i}, shape);
            m = m + size(A_cell{i},1);
        end
        
        if m == 0
            A = zeros(1,n);
            c = zeros(1,1);
            return;
        end
        
        A = zeros(m,n);
        c = zeros(m,1);
        i_row = 1;
        for i=1:n_functions
            A(i_row:i_row+size(A_cell{i},1)-1,:) = A_cell{i};
            c(i_row:i_row+size(A_cell{i},1)-1) = c_cell{i};
            i_row = i_row + size(A_cell{i},1);
        end
    end
end

function [A,c] = calc_single_lin_exp(f, shape)
    if shape(1)~=1 && shape(2)~=1
        error('shape of variable should be a vector');
    end
    if shape(1)~=1
        n = shape(1);
    else
        n = shape(2);
    end
    
    x_sym = sym('x', [n 1]);
    
    obj = f(x_sym);
    obj = reshape(obj, [], 1);
    m = size(obj,1);

    A = zeros(m,n);
    c = zeros(m,1);
    
    for i=1:m
        a_i = jacobian(obj(i), x_sym);
        c_i = obj(i) - a_i*x_sym;

        err_i = obj(i) - (a_i*x_sym + c_i);
        if err_i ~= 0
            error('unexpected error in expressing function in linear form');
        end
        
        try
            A(i,:) = double(a_i);
            c(i) = double(c_i);
        catch err
            error('given function cannot be expressed in linear form');
        end
    end
    
end
