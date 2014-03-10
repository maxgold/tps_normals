close all
clear

paren = @(x, varargin) x(varargin{:});

addpath ../q2_starter/  %TODO: your path might be different depending on where you solved q2

T = 30;
K = 3;
getTraj = @(x) reshape(x, K, T);

traj_init = [linspace(-pi/2,pi/2,T);...
             zeros(1,T);...
             zeros(1,T)];

obstacles = {orientedBoxToPolygon([1.6 0 1 .5 135]), ...
    orientedBoxToPolygon([1 -1.5 1 .5 45])};

% The trajectory is a 3xT matrix
% the rows are the degrees of freedom of the robot: x,y,angle.
% the columns are timesteps 1 to T.

% We will flatten this matrix for optimization in the column-major
% way: x = traj_init(:) = (x_1, y_1, theta_1, x_2, y_2, theta_2, ...)
% Keep that in mind when defining the cost and constraints below.

x0 = traj_init(:);

dsafe = 0.05; % Safety margin
KT = K*T;

robot_lengths = [0.8 0.6 0.5];

% Function that maps state vector to polygon (which is used for collision
% checking)
make_robot_poly = @(x) make_robot_poly_linky(robot_lengths, x);


%q_m = zeros(1,KT);

% Create Q matrix for the sum-of-squared-displacements cost
% \sum_t || \theta_{t+1} - \theta_t ||^2
% where \theta is the state vector (x, y, angle)

quad_obj = @(x) sum(sum(square(paren(getTraj(x),:,2:T) - paren(getTraj(x),:,1:T-1))));
[Q, q, c] = calc_quad_expr(quad_obj, [KT,1]);

f = @(x) 0;
% The constraint function g does all the work of computing signed distances
% and their gradients
g = @(x) g_collisions_linky(robot_lengths, x, dsafe, [K,T], make_robot_poly, obstacles);
h = @(x) 0;

% Create linear inequality constraints to enforce that the
% displacement is smaller than .2 (componentwise) over each timestep

f_ineqs = { ...
    @(x) paren(getTraj(x),:,2:T) - paren(getTraj(x),:,1:T-1) - .2*ones(K,T-1), ...
    @(x) paren(getTraj(x),:,1:T-1) - paren(getTraj(x),:,2:T) - .2*ones(K,T-1) ...
    };
[A_ineq, neg_b_ineq] = calc_lin_expr(f_ineqs, [KT,1]);
b_ineq = -neg_b_ineq;

% Create linear equality constraints to fix the beginning and end of the
% trajectory to the values in traj_init(:,1) and traj_init(:,end)

f_eqs = { ...
    @(x) paren(getTraj(x),:,1) - traj_init(:,1), ...
    @(x) paren(getTraj(x),:,T) - traj_init(:,T) ...
    };
[A_eq, neg_b_eq] = calc_lin_expr(f_eqs, [KT,1]);
b_eq = -neg_b_eq;

cfg = struct();
cfg.callback = @(x,~) plot_traj(make_robot_poly, obstacles, reshape(x,size(traj_init)));
cfg.initial_trust_box_size=.1;
cfg.g_use_numerical = false;
cfg.min_approx_improve = 1e-2;

x = penalty_sqp(x0, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg);