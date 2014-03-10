close all
%clear

paren = @(x, varargin) x(varargin{:});
curly = @(x, varargin) x{varargin{:}};

[X_s, X_s_new, X_g, obstacles, obstacles_new] = generate_tunnel_ex();
[d,n] = size(X_s);

% TPS
K = tps_kernel(X_s);
N = null([X_s; ones(1,n)]);
tps_dim = (n-(d+1))*d + d*d + d; % dim(A) + dim(B) + dim(c)

% TODO: parameters to tune
lambda = 0.1; % TPS regularization coefficient
alpha = 4; % penalty coefficient for the TPS transformation
beta = 1; % penalty coeffient for trajectory points deviating from the transformed demonstration trajectory

% The robot trajectory is a state_dim x T matrix
% the rows are the degrees of freedom of the robot: x,y,angle.
% the columns are timesteps 1 to T.
[state_dim, T] = size(X_g);
state_dim_T = state_dim * T;

dsafe = 0.05; % Safety margin
joint_limit_min = zeros(state_dim, 1);
robot_angles = deg2rad([0;90;-90;-90;90]);
robot_width = 0.2;

% Function that maps state vector to polygon (which is used for collision
% checking)
make_robot_poly = @(x) make_robot_poly_scopey(robot_angles, robot_width, x);

% Functions for extracting trajectory and TPS transformation parameters
% from the variable being optimized
% sequence of states
getTraj = @(x) reshape(x(1:state_dim_T), state_dim, T);
% sequence of states' points. these points are 2D points on the geometry of the robot for every state
getTrajPts = @(x) get_traj_pts_scopey(getTraj, robot_angles, x);
% A B c describes the TPS transformation
getA = @(x) N*reshape(x(state_dim_T+1:state_dim_T+(n-(d+1))*d), (n-(d+1)), d);
getB = @(x) reshape(x(state_dim_T+(n-(d+1))*d+1:end-d), d, d);
getc = @(x) x(end-d+1:end);

% The variable being optimized is the flattened concatenation of the
% trajectory and the TPS transformation parameters.
% x = [traj(:); A(:); B(:); c(:)]

[~, iwarp, ~, Q_tps, B_tps, C_tps] = compute_warp(X_s, X_s_new, lambda);
% The new x0 uses the warp function found by TPS as the initialization
x0 = [X_g(:); Q_tps(:); B_tps(:); C_tps];

% Below is old x0
%x0 = [X_g(:); zeros((n-(d+1))*d,1); reshape(eye(d),d*d,1); zeros(d,1)];

% create Q matrix for the sum-of-squared-displacements cost
% \sum_t || \theta_{t+1} - \theta_t ||^2 
%   + alpha*(||X_s_new - f(X_s)||^2 + \lambda * regularizer(f))
%   + beta*\sum_t ||points(\theta_t) - f(points(X_g_t))||^2
% where \theta is the state vector (x, y, angle)
% and points maps the state vector into points (which are on the
% geometry of the robot at that state)
quad_obj = @(x) sum(sum(square(paren(getTraj(x),:,2:T) - paren(getTraj(x),:,1:T-1)))) + ...
    alpha*(sum(sum(square(X_s_new' - K*getA(x) - X_s'*getB(x) - ones(n,1)*getc(x).'))) + lambda * trace(getA(x).'*K*getA(x))) + ...
    beta*(sum(sum(square(getTrajPts(x) - warp_pts(getTrajPts(X_g), make_warp(getA(x), getB(x), getc(x), X_s))))));
% Commenting out [Q, q] for efficiency
[Q, q] = calc_quad_expr(quad_obj, [state_dim_T + tps_dim,1]);
f = @(x) 0;

% The constraint function g does all the work of computing signed distances
% and their gradients
%obstacles_new2 = {obstacles_new{1:4}, obstacles_new{6}};
g = @(x) g_collisions_scopey(robot_angles, x, dsafe, [state_dim,T], make_robot_poly, obstacles_new);
h = @(x) 0;

% create linear inequality constraints to enforce that the
% displacement is smaller than .2 (componentwise) over each timestep
f_ineqs = { ...
    %@(x) paren(getTraj(x),:,2:T) - paren(getTraj(x),:,1:T-1) - .2*ones(state_dim,T-1), ...
    %@(x) paren(getTraj(x),:,1:T-1) - paren(getTraj(x),:,2:T) - .2*ones(state_dim,T-1) ...
    };

for t=1:T
    for k=1:state_dim
        f_ineqs{end+1} = @(x) joint_limit_min(k) - paren(getTraj(x),k,t);
    end
end

[A_ineq, neg_b_ineq] = calc_lin_expr(f_ineqs, [state_dim_T + tps_dim,1]);
b_ineq = -neg_b_ineq;

% create linear equality constraints to fix the beginning and end of the
% trajectory to the beginning and end of the warped demonstrated trajectory
f_eqs = { ...
    @(x) paren(getTrajPts(x),:,1) - X_s_new(:,end-1), ...
    @(x) paren(getTrajPts(x),:,T) - X_s_new(:,end), ...
    @(x) ones(1,n)*getA(x) ...
    };
[A_eq, neg_b_eq] = calc_lin_expr(f_eqs, [state_dim_T + tps_dim,1]);
b_eq = -neg_b_eq;

orig_fig = figure();
figure(orig_fig);
hold on;
plot_traj(make_robot_poly, {}, X_g, false, [-0.5 6.5 -2 2]);
h1 = scatter(X_s(1,:), X_s(2,:), 'red');
h2 = scatter(X_g(1,:), X_g(2,:), 'cyan', 'x');
draw_grid_orig([-0.5 2], [6.5 -2], 5);
legend([h1 h2], 'x_i^{(S)}', 'x_t^{(G)}');

warp_fig = figure();
figure(warp_fig);

cfg = struct();
cfg.callback = @(x,~) plot_traj_warp(make_warp(getA(x), getB(x), getc(x), X_s), X_s_new, X_s, getTraj(x), X_g, getTrajPts, @(X) plot_traj(make_robot_poly, obstacles_new, X, false, [-0.5 6.5 -2 2]), [-0.5 2 6.5 -2]);
cfg.initial_trust_box_size=.1;
cfg.g_use_numerical = false;
cfg.min_approx_improve = 1e-2;

x = penalty_sqp(x0, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg);
%   minimize (1/2) x'*Q*x + q*x + f(x)
%   subject to
%       A_ineq*x <= b_ineq
%       A_eq*x == b_eq
%       g(x) <= 0
%       h(x) == 0

warp = make_warp(getA(x), getB(x), getc(x), X_s);
plot_traj_warp(warp, X_s_new, X_s, getTraj(x), X_g, getTrajPts, @(X) plot_traj(make_robot_poly, obstacles_new, X, true, [-0.5 6.5 -2 2]), [-0.5 2 6.5 -2]);

