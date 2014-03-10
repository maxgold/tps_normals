function U_t = der_U(z,t)
    r = norm(z);
    if r == 0
        U_t = 0;
    else
        U_t = (2*log(r)+1)*(z'*t);
    end
end
