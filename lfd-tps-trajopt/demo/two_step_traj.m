load 'two_step_scopey_lambda1.mat';

plot_traj_warp(warp_fn, X_s_new, X_s, result_traj, X_g, getTrajPts, @(X) plot_traj(make_robot_poly, obstacles_new, X, true, [-0.5 6.5 -2 2]), [-0.5 2 6.5 -2]);
