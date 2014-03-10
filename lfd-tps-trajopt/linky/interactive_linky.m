close all
clear

paren = @(x, varargin) x(varargin{:});
curly = @(x, varargin) x{varargin{:}};

T = 2;
K = 4;
KT = K*T;
traj_init = zeros(K,T);
x0 = traj_init(:);
getTraj = @(x) reshape(x, K, T);

obstacles = { ...
    orientedBoxToPolygon([0 -2.25 4 .5 0]), ...
    orientedBoxToPolygon([0.8 .5 .5 .5]), ...
    orientedBoxToPolygon([1.9 .5 .5 .5]), ...
    orientedBoxToPolygon([0.8 1.25 .5 .5]), ...
    orientedBoxToPolygon([1.9 1.25 .5 .5]), ...
    };

dsafe = 0.05; % Safety margin

robot_lengths = [0.8 0.8 0.8 0.6];
make_robot_poly = @(x) make_robot_poly_linky(robot_lengths, x);

% quadratic objective
quad_obj = @(x) sum(sum(square(paren(getTraj(x),:,2:T) - paren(getTraj(x),:,1:T-1))));
[Q, q, c] = calc_quad_expr(quad_obj, [KT,1]);
% non-linear objective
f = @(x) 0;
    
% non-linear inequality constraints
g = @(x) g_collisions_linky(robot_lengths, x, dsafe, [K,T], make_robot_poly, obstacles);

% linear inequality constraints
f_ineqs = {};
[A_ineq, neg_b_ineq] = calc_lin_expr(f_ineqs, [KT,1]);
b_ineq = -neg_b_ineq;

cfg = struct();
%cfg.callback = @(x,~) plot_traj(make_robot_poly, obstacles, getTraj(x));
cfg.initial_trust_box_size=.1;
cfg.f_use_numerical = true;
cfg.g_use_numerical = false;
cfg.h_use_numerical = false;
cfg.min_approx_improve = 1e-2;

while true
    plot_traj(make_robot_poly, obstacles, paren(getTraj(x0),:,T), false, [-4 4 -4 4]);

    pt_target = ginput(1)';
    pt_handle = plot(pt_target(1), pt_target(2), 'x');
    pause(0.01);
    
    % non-linear equality constraints
    h = @(x) kinematic_cnt_linky(robot_lengths, x, [K,T], T, pt_target);

    % linear equality constraints
    f_eqs = { ...
        @(x) paren(getTraj(x),:,1) - paren(getTraj(x0),:,T), ...
        };
    [A_eq, neg_b_eq] = calc_lin_expr(f_eqs, [KT,1]);
    b_eq = -neg_b_eq;

    x1 = penalty_sqp(x0, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg)
    
    delete(pt_handle);
    
    x0 = x1;
end
