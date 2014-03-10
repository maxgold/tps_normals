load 'tps_trajopt_scopey.mat';

plot_traj_warp(warp, X_s_new, X_s, getTraj(x), X_g, getTrajPts, @(X) plot_traj(make_robot_poly, obstacles_new, X, true, [-0.5 6.5 -2 2]), [-0.5 2 6.5 -2]);
