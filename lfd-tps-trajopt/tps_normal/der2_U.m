function U_titj = der2_U(z,ti,tj)
    r = norm(z);
    if r == 0
        U_titj = 0;
    else
        U_titj = 2*(z'*ti)*(z'*tj)/(r^2) + (2*log(r)+1)*(tj'*ti);
    end
end
