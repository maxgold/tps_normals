function k = kernel_vector(warp_pt, data_array)
  [d, n] = size(data_array);
  k = zeros(n, 1);
  for i = 1:n
    r = norm(warp_pt - data_array(:,i));
    if r == 0
        k(i) = 0;
    else
        if d==2 || d==4
            k(i) = (r^(4-d))*log(r);
        else
            k(i) = -r^(4-d);
        end
    end
  end
end
