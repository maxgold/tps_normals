function jac = calc_jac_scopey(pt, angles, x)
    if size(angles,1) ~= 1
        angles = angles';
    end
    
    cs = cumsum(angles);
    jac = [cos(cs); sin(cs)];
end