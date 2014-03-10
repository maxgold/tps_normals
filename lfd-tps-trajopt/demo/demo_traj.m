load 'tps_trajopt_scopey.mat';

figure(orig_fig);
hold on;
plot_traj(make_robot_poly, obstacles, X_g, true, [-0.5 6.5 -2 2]);

X_g = getTrajPts(X_g);
h1 = scatter(X_s(1,:), X_s(2,:), 50, 'red', 'LineWidth', 2);
h2 = scatter(X_g(1,:), X_g(2,:), 50, 'cyan', 'x', 'LineWidth', 2);
draw_grid_orig([-0.5 2], [6.5 -2], 5);
hl = legend([h1 h2], '$\mathbf{x}_s^{\emph{(i)}}$', '$\mathbf{\phi}_s^{\emph{(j)}}$');
set(hl,'Interpreter','latex')
