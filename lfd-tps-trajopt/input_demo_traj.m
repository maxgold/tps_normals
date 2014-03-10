function demo_traj = input_demo_traj(X, Y, demo_obs, test_obs, dim)
  success = false;
  while ~success
    close all;
    warp_fn = plot_current_warp(X, Y, demo_obs, test_obs, dim);
    figure(1);
    disp('Enter Demo Trajectory');
    demo = ginput();
    test = warp_pts(demo', warp_fn)';
    figure(1);
    scatter(demo(:,1), demo(:,2));
    figure(2);
    scatter(test(:,1), test(:,2));
    success = input('Enter 0 to try a different trajecotry');
  end
  demo_traj = demo';
end

function warp_fn = plot_current_warp(X, Y, demo_obs, test_obs, dim)
  [~, warp_fn, ~] = compute_warp(X, Y);
  clf;
  demo_fig = figure(1);
  test_fig = figure(2);
  upper_left = dim(1:2);
  bottom_right = dim(3:4);
  draw_grid(upper_left, bottom_right, warp_fn, 10, demo_fig, test_fig);
  figure(1);
  hold on;
  scatter(X(:,1), X(:,2));
  for obs_cell = demo_obs
    obs = obs_cell{1};
    plot([obs(:,1)',obs(1,1)],[obs(:,2)',obs(1,2)]);
  end  
  figure(2);
  hold on;
  scatter(Y(:,1), Y(:,2));
  for obs_cell = test_obs
    obs = obs_cell{1};
    plot([obs(:,1)',obs(1,1)],[obs(:,2)',obs(1,2)]);
  end  
end
