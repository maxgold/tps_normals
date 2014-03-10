%%%%%%%%%%%%%%
% 

l = [0.5;0.5;0.5];
x = deg2rad([30;45;-60]);
traj_shape = [3,1];
ts = [1];
target_pts = [1.25; 1];
ftest = @(x) norm(fwd_kin_linky(l,x) - target_pts,2)^2;
[grad, hess] = numerical_grad_hess(ftest, x, true);
[y,grad_m, hess_m] = kinematic_cost_linky(l, x, traj_shape, ts, target_pts);
% At this point: Should satisfy grad = grad_m, hess - hess_m

f_kin_cnt = @(x) kinematic_cnt_linky(l,x,traj_shape,ts,target_pts);
grad = numerical_jac(f_kin_cnt, x);
[y,grad_m] = kinematic_cnt_linky(l,x,traj_shape,ts,target_pts);
% Checkpoint: grad should equal grad_m

%%%%%%%%%%%
% check gradient and hessian for kinematic_cost_scopey
T = 5;
X_g = X_g(:,1:T);
target_pts = target_pts(:,1:T);

x = X_g(:);
traj_shape = [K,T];
ts = 1:T;
[y, grad_m, hess_m] = kinematic_cost_scopey(robot_angles, x, traj_shape, ts, target_pts);

getTraj = @(x) reshape(x, K, T);
getTrajPts = @(x) get_traj_pts_scopey(getTraj, robot_angles, x);
f_kin_cost_scopey = @(x) sum(sum(square(getTrajPts(x) - target_pts)));
[grad, hess] = numerical_grad_hess(f_kin_cost_scopey, x, true);

%%%%%%%%%%%%
% Testing analytical Jacobian for Scopey

angles = deg2rad([45; 45; -90; -90]);
robot_width = 0.2;
obstacles = {orientedBoxToPolygon([2 0 2 1 0])};
x = 0.5*ones(4,1);
dsafe = 0.2;
traj_shape = [4,1];
make_robot_poly = @(x) make_robot_poly_scopey(angles, robot_width, x);

[val, jac_m] = g_collisions_scopey(angles, x, dsafe, traj_shape, make_robot_poly, obstacles);

g = @(x) g_collisions_scopey(angles, x, dsafe, traj_shape, make_robot_poly, obstacles);
jac = numerical_jac(g, x);
% Checkpoint: jac_m should equal jac