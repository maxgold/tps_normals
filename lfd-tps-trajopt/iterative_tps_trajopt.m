close all
clear
addpath ./q2_starter
addpath ./q3_starter
% scene correspondences
X_s = [1 1; 1 -1; -1 -1; -1 1]';
X_s_new = [1 2; 1.2 -.8; -1 -1; -1 1.5]';
obstacles = {orientedBoxToPolygon([0 0 1 1 0])};
N = size(X_s, 2);
T = 20;
X_g = [ linspace(-1,1,T); zeros(1,T); zeros(1,T) ];
X_g_pts = X_g(1:2,:);
X_s_g = [X_s, X_g(1:2,:)];

[warped_pts, warp_fn, ~] = compute_warp(X_s, X_s_new);
target_traj = [warp_pts(X_g_pts, warp_fn); X_g(3,:)];

car_length = .4;
car_width = .2;
make_robot_poly = @(x) orientedBoxToPolygon([x(1), x(2), car_length, car_width, rad2deg(x(3))]);

feasible_traj = follow_trajectory(target_traj, obstacles, make_robot_poly);
traj_points = feasible_traj(1:2,:);
traj_theta = feasible_traj(3,:);
old_traj = -1*feasible_traj(:,:)
stopping_eps = 1e-2
while norm(old_traj - feasible_traj, 2) > stopping_eps
      err = norm(old_traj - feasible_traj, 2)
      pause(2)
      X_s_g_new = [X_s_new, traj_points];
      [warped_pts, warp_fn, ~] = compute_warp(X_s_g, X_s_g_new);
      size(warped_pts(:,N+1:end))
      size(traj_theta)
      target_traj = [warped_pts(:, N+1:end); traj_theta];
      old_traj = feasible_traj;
      feasible_traj = follow_trajectory(target_traj, obstacles, make_robot_poly);
      traj_points = feasible_traj(1:2,:);
      traj_theta = feasible_traj(3,:);
      plot_traj_warp(warp_fn, X_s_new, X_s, feasible_traj, X_g, @(x) x(1:2,:), @(X) plot_traj(make_robot_poly, obstacles, X, false));      

end
