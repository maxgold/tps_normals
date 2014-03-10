function warp = make_warp_normal(w, w_til, X_s, Xe_s, Xn_s)
  warp = @(warp_pt) compute_warp(warp_pt, w, w_til, X_s, Xe_s, Xn_s);
end

function pt = compute_warp(warp_pt, w, w_til, X_s, Xe_s, Xn_s)
    [~,m] = size(Xn_s);
    slope_fun = zeros(m, 1);
    for i = 1:m
        slope_fun(i) = -der_U(warp_pt - Xe_s(:,i), Xn_s(:,i));
    end
    pt = [kernel_vector(warp_pt, X_s); 1; warp_pt; slope_fun]'*[w; w_til];
end