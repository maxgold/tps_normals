function plot_traj_warp(warp, X_s_new, X_s, X_g_new, X_g, getPts, plot_traj, varargin)
    % varargin can only be [upper_left_x upper_left_y bottom_right_x bottom_right_y]
    
    clf;
    hold on;
    
    plot_traj(X_g_new);
    
    upper_left = [-1 1];
    bottom_right = [1 -1];
    if ~isempty(varargin)
        upper_left = varargin{1}(1:2);
        bottom_right = varargin{1}(3:4);
    end
    
    draw_grid_warp(upper_left, bottom_right, warp, 5);

    X_s_warped = warp_pts(X_s, warp);
    X_g_new = getPts(X_g_new);
    X_g_warped = warp_pts(getPts(X_g), warp);
    h1 = scatter(X_s_new(1,:), X_s_new(2,:), 50, 'green', 'LineWidth', 2);
    h2 = scatter(X_s_warped(1,:), X_s_warped(2,:), 50, 'red', 'LineWidth', 2);
    h3 = scatter(X_g_new(1,:), X_g_new(2,:), 50, 'magenta', 'LineWidth', 2);
    h4 = scatter(X_g_warped(1,:), X_g_warped(2,:), 50, 'cyan', 'x', 'LineWidth', 2);
    hl = legend([h1 h2 h3 h4], '$\mathbf{x}_t^{\emph{(i)}}$', '$f(\mathbf{x}_s^{\emph{(i)}})$', '$\mathbf{\phi}_t^{\emph{(j)}}$', '$f(\mathbf{\phi}_s^{\emph{(j)}})$');
    set(hl,'Interpreter','latex')

    pause(.02);
end
