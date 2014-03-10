function result = follow_trajectory(target, obstacles, make_robot_poly)
         addpath ./q2_starter
	 addpath ./q3_starter
	 [K, T] = size(target);
	 x0 = target(:);
	 dsafe = 0.05; % Safety margin
	 KT = K*T;
	 
	 q = zeros(1,KT);
	 Q = 0;
	 upper_right = -1*[zeros(KT-K, K), eye(KT-K);
			   zeros(K, KT)];
	 Q = 2*(eye(KT) + upper_right);
	 Q(1:K, 1:K) = eye(K);
	 Q(end-K+1:end, end-K+1:end) = eye(K);

	 f = @(x) .1*norm(x-x0, 2)^2;

	 g = @(x) g_collisions(x, dsafe, [K,T], make_robot_poly, obstacles);
	 h = @(x) 0;

	 A_ineq =  eye(KT) + upper_right;
	 A_ineq = A_ineq(1:end-K, :);
	 A_ineq = [A_ineq;-1*A_ineq];%% include x1 - x2 and x2-x1
	 b_ineq =  .5*ones(2*(KT-K), 1);

%	 A_eq = zeros(2*K,KT);
%	 A_eq(1:K,1:K) = eye(K);
%	 A_eq(K+1:2*K,KT-K+1:KT) = eye(K);
%	 b_eq = [target(:,1); target(:,end)];
	 A_eq = 0;
	 b_eq = 0;

	 cfg = struct();
	 cfg.callback = @(x,~) plot_traj(make_robot_poly, obstacles, reshape(x,size(target)));
	 cfg.initial_trust_box_size=.1;
	 cfg.min_trust_box_size = 1e-2;
	 cfg.g_use_numerical = false;
	 cfg.min_approx_improve = 1e-2;


	 result = reshape(penalty_sqp(x0, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg), size(target));

end
