function jac = calc_jac_linky(pt, l, x)
    if size(pt,2) ~= 1
        pt = pt';
    end
    if size(l,1) ~= 1
        l = l';
    end
    if size(x,1) ~= 1
        x = x';
    end
    r = pt - fwd_kin_linky(l(1:end-1), x(1:end-1)); % pt relative to second to last link
    cs = cumsum(x(1:end-1));
    jac = fliplr(cumsum(fliplr([-l(1:end-1).*sin(cs); l(1:end-1).*cos(cs)]),2));
    jac = [jac zeros(2,1)] + repmat([-r(2); r(1)], 1, size(jac,2)+1);
end