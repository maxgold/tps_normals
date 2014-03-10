function [warped_pts, warp_fn, err, Q, B, c] = compute_warp(X, Y, varargin)
% are initial points
% Y are cooresponding points we would like to match to
     lambda = 1;
     if ~isempty(varargin)
         lambda = varargin{1};
     end
	 [d,n] = size(X);
	 K = tps_kernel(X);
	 N = null([X; ones(1,n)]);
     
     try
         sqrtKN = chol(N'*K*N);
     catch err
         % workaround of numerical issues to deal with tmp that is almost not psd
         tmp = N'*K*N;
         tmp = (tmp'+tmp)/2;
         while min(eig(tmp)) < 0
             warning('N^T K N is not positive semidefinite... adjusting it');
             tmp = tmp - eye(size(tmp))*4*min(eig(tmp));
         end
         sqrtKN = chol(tmp);
     end
     
	 cvx_begin quiet
	       variables Q(n-(d+1),d) B(d,d) c(d) s;
	       
	       minimize (pow_pos(norm(Y' - K*N*Q - X'*B - ones(n,1)*c'),2) + lambda * sum(sum_square(sqrtKN*Q)))
	       % equivalent ton
	       % minimize (norm(X_s_new' - K*A - X_s'*B - ones(n,1)*c')^2 + lambda * trace(A'*K*A))
	       
	       subject to
	       ones(1, n)*(N*Q) == zeros(1, d);
	 cvx_end

	 warp_fn = make_warp(N*Q, B, c, X);
	 warped_pts = warp_pts(X, warp_fn);
	 err = norm(Y - warped_pts, 2)^2;
	 
end
