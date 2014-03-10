function warp = make_warp(A, B, c, data_array)
  %MAKE_WARP Constructs a warping function given its parameters
  %
  % make_warp(A, B, c, data_array)
  %
  % The warping function that is returned takes only one parameter, the point
  % (column vector) to be warped, and returns a column vector representing the
  % point after warping.
  %
  % Note that this function does not find the parameters for a warp -- generally
  % finding the parameters involves solving an optimization problem of the
  % correct form.
  %
  % See the LFD paper for more details.

  warp = @(warp_pt) compute_warp(warp_pt, A, B, c, data_array);
end

function pt = compute_warp(warp_pt, A, B, c, data_array)
  k = kernel_vector(warp_pt, data_array);
  pt = A.' * k + B.' * warp_pt + c;
end
  
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
