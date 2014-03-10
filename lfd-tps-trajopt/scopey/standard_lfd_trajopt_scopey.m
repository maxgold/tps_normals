[X_s, X_s_new, X_g, obstacles, obstacles_new] = generate_tunnel_ex();

robot_angles = deg2rad([0;90;-90;-90;90]);
robot_width = 0.2;
make_robot_poly = @(x) make_robot_poly_scopey(robot_angles, robot_width, x);
[K, T] = size(X_g);
getTraj = @(x) reshape(x(1:K*T), K, T);
getTrajPts = @(x) get_traj_pts_scopey(getTraj, robot_angles, x);

lambda = 0.1;
[~, warp_fn, ~] = compute_warp(X_s, X_s_new, lambda);

target_pts = warp_pts(getTrajPts(X_g), warp_fn);

start_pt = X_s_new(:,end-1);
end_pt = X_s_new(:,end);
result_traj = follow_trajectory_scopey(X_g, target_pts, start_pt, end_pt, obstacles_new, make_robot_poly, robot_angles, getTraj, getTrajPts);

plot(target_pts(1,:), target_pts(2,:), 'cx');

result_pts = getTrajPts(result_traj);

%[~, ~, warp_cost] = compute_warp([X_s, getTrajPts(X_g)], [X_s_new, result_pts]);
%warp_cost
