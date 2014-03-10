function p = fwd_kin_linky(l,x)
% x = angles
% l = lengths
    if size(x,1) ~= 1
        x = x.';
    end    
    if size(l,1) ~= 1
        l = l.';
    end
    cs = cumsum(x);
    p = sum([l.*cos(cs); l.*sin(cs)], 2);
end