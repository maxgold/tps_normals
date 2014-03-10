function [result_pts, warp_cost] = standard_lfd_trajopt(X, Y, demo_obs, test_obs, demo_pts)
  car_length = .4;
  car_width = .2;
  make_robot_poly = @(x)orientedBoxToPolygon([x(1),x(2),car_length,car_width,rad2deg(x(3))]);
  [~, warp_fn, ~] = compute_warp(X, Y)
  target_pts = warp_pts(demo_pts, warp_fn);
  target_traj = [target_pts;zeros(1, size(target_pts, 2))];
  result_traj = follow_trajectory(target_traj, test_obs, make_robot_poly);
  result_pts = result_traj(1:2, :);
  [~, ~, warp_cost] = compute_warp([X, demo_pts], [Y, result_pts]);
end