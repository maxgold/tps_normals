function traj = follow_trajectory_scopey(traj_init, target_pts, start_pt, end_pt, obstacles, make_robot_poly, robot_angles, getTraj, getTrajPts)
    paren = @(x, varargin) x(varargin{:});
    curly = @(x, varargin) x{varargin{:}};

    [K, T] = size(traj_init);
    x0 = traj_init(:);
    dsafe = 0.05; % Safety margin
    KT = K*T;
    joint_limit_min = zeros(K, 1);
    
    quad_obj = @(x) sum(sum(square(paren(getTraj(x),:,2:T) - paren(getTraj(x),:,1:T-1))));
    [Q, q] = calc_quad_expr(quad_obj, [KT,1]);

    f = @(x) 0;
    %f = @(x) kinematic_cost_scopey(robot_angles, x, [K,T], 1:T, target_pts);
    
    % The constraint function g does all the work of computing signed distances
    % and their gradients
    g = @(x) g_collisions_scopey(robot_angles, x, dsafe, [K,T], make_robot_poly, obstacles);
    %h = @(x) 0;
    h = @(x) kinematic_cnt_scopey(robot_angles, x, [K,T], 1:T, target_pts);

    
    % Create linear inequality constraints to enforce that the
    % displacement is smaller than .2 (componentwise) over each timestep
    f_ineqs = { ...
        %@(x) paren(getTraj(x),:,2:T) - paren(getTraj(x),:,1:T-1) - .2*ones(K,T-1), ...
        %@(x) paren(getTraj(x),:,1:T-1) - paren(getTraj(x),:,2:T) - .2*ones(K,T-1) ...
        };
    for t=1:T
        for k=1:K
            f_ineqs{end+1} = @(x) joint_limit_min(k) - paren(getTraj(x),k,t);
        end
    end
    [A_ineq, neg_b_ineq] = calc_lin_expr(f_ineqs, [KT,1]);
    b_ineq = -neg_b_ineq;

    % Create linear equality constraints to fix the beginning and end of the
    % trajectory to the values in traj_init(:,1) and traj_init(:,end)
    f_eqs = { ...
        @(x) paren(getTrajPts(x),:,1) - start_pt, ...
        @(x) paren(getTrajPts(x),:,T) - end_pt, ...
        };
    [A_eq, neg_b_eq] = calc_lin_expr(f_eqs, [KT,1]);
    b_eq = -neg_b_eq;

    cfg = struct();
    cfg.callback = @(x,~) plot_traj(make_robot_poly, obstacles, reshape(x,size(traj_init)), false, [-1,6,-2,2]);
    cfg.initial_trust_box_size=.1;
    cfg.h_use_numerical = false;
    cfg.g_use_numerical = false;
    cfg.min_approx_improve = 1e-2;

    x = penalty_sqp(x0, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg);
    traj = getTraj(x);
end
