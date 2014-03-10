function jac = numerical_jac_linky(pt, l, x)
    if size(pt,2) ~= 1
        pt = pt';
    end
    x_dim = length(x);
    
    pt = [pt; 0];
    unit_z = [0 0 1]';
    jac = zeros(3, x_dim);
    for j=1:x_dim
        if j==1
            p = zeros(3,1);
        else
            p = [fwd_kin_linky(l(1:j-1), x(1:j-1)); 0];
        end
        jac(:,j) = cross(unit_z, pt-p);
    end
    jac = jac(1:2,:);
end