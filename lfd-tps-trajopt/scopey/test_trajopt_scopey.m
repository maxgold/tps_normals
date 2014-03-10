close all
clear

paren = @(x, varargin) x(varargin{:});

%addpath ../q2_starter/  %TODO: your path might be different depending on where you solved q2

T = 30;
K = 4;
getTraj = @(x) reshape(x, K, T);
robot_width = 0.2;

joint_limit_max = 5*ones(K, 1);
joint_limit_min = 0.1*ones(K, 1);
robot_angles = deg2rad([45; 45; -90; -90]);

%traj_init = [linspace(0.5,4.5,T);...
traj_init = [0.5*ones(1,T);...
             0.5*ones(1,T);...
             0.5*ones(1,T);...
             0.5*ones(1,T)];
pt_target = [5;0];

%obstacles = {orientedBoxToPolygon([1.6 0 1 .5 135]), ...
%    orientedBoxToPolygon([1 -1.5 1 .5 45])};
obstacles = {orientedBoxToPolygon([2 0 2 1 -30])};
%obstacles = {};

% The trajectory is a 3xT matrix
% the rows are the degrees of freedom of the robot: x,y,angle.
% the columns are timesteps 1 to T.

% We will flatten this matrix for optimization in the column-major
% way: x = traj_init(:) = (x_1, y_1, theta_1, x_2, y_2, theta_2, ...)
% Keep that in mind when defining the cost and constraints below.

x0 = traj_init(:);

dsafe = 0.05; % Safety margin
KT = K*T;

% Function that maps state vector to polygon (which is used for collision
% checking)
make_robot_poly = @(x) make_robot_poly_scopey(robot_angles, robot_width, x);


%q_m = zeros(1,KT);

% Create Q matrix for the sum-of-squared-displacements cost
% \sum_t || \theta_{t+1} - \theta_t ||^2
% where \theta is the state vector (x, y, angle)

quad_obj = @(x) sum(sum(square(paren(getTraj(x),:,2:T) - paren(getTraj(x),:,1:T-1))));
[Q, q, c] = calc_quad_expr(quad_obj, [KT,1]);

f = @(x) 0;
% The constraint function g does all the work of computing signed distances
% and their gradients
g = @(x) g_collisions_scopey(robot_angles, x, dsafe, [K,T], make_robot_poly, obstacles);
h = @(x) kinematic_cnt_scopey(robot_angles, x, [K,T], T, pt_target);

% Create linear inequality constraints to enforce that the
% displacement is smaller than .2 (componentwise) over each timestep

f_ineqs = { ...
    %@(x) paren(getTraj(x),:,2:T) - paren(getTraj(x),:,1:T-1) - .2*ones(K,T-1), ...
    %@(x) paren(getTraj(x),:,1:T-1) - paren(getTraj(x),:,2:T) - .2*ones(K,T-1) ...
    };

for t=1:T
    for k=1:K
        f_ineqs{end+1} = @(x) paren(getTraj(x),k,t) - joint_limit_max(k);
        f_ineqs{end+1} = @(x) joint_limit_min(k) - paren(getTraj(x),k,t);
    end
end

[A_ineq, neg_b_ineq] = calc_lin_expr(f_ineqs, [KT,1]);
b_ineq = -neg_b_ineq;

% Create linear equality constraints to fix the beginning and end of the
% trajectory to the values in traj_init(:,1) and traj_init(:,end)

f_eqs = { ...
    @(x) paren(getTraj(x),:,1) - traj_init(:,1)%, ...
    %@(x) paren(getTraj(x),:,T) - traj_init(:,T) ...
    };
[A_eq, neg_b_eq] = calc_lin_expr(f_eqs, [KT,1]);
b_eq = -neg_b_eq;

cfg = struct();
cfg.callback = @(x,~) plot_traj(make_robot_poly, obstacles, reshape(x,size(traj_init)), [-1,6,-2,2]);
cfg.initial_trust_box_size=.1;
cfg.g_use_numerical = false;  %TODO - make our analytical jacobian more accurate
cfg.min_approx_improve = 1e-2;

x = penalty_sqp(x0, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg);