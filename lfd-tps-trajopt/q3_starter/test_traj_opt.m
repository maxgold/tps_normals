close all
clear
setup

addpath ../q2_starter/  %TODO: your path might be different depending on where you solved q2

T = 40;
K = 3;
traj_init = [linspace(-2,2,T);...
             zeros(1,T);...
             zeros(1,T)];

         
load obstacles1

% The trajectory is a 3xT matrix
% the rows are the degrees of freedom of the robot: x,y,angle.
% the columns are timesteps 1 to T.

% We will flatten this matrix for optimization in the column-major
% way: x = traj_init(:) = (x_1, y_1, theta_1, x_2, y_2, theta_2, ...)
% Keep that in mind when defining the cost and constraints below.

x0 = traj_init(:);

dsafe = 0.05; % Safety margin
KT = K*T;

car_length = .4;
car_width = .2;

% Function that maps state vector to polygon (which is used for collision
% checking)
make_robot_poly = @(x) orientedBoxToPolygon([x(1), x(2), car_length, car_width, rad2deg(x(3))]);


q = zeros(1,KT);

% TODO: create Q matrix for the sum-of-squared-displacements cost
% \sum_t || \theta_{t+1} - \theta_t ||^2
% where \theta is the state vector (x, y, angle)

%YOUR_CODE_HERE;
%{
Q = eye(KT);
Q(1+K:KT-K,1+K:KT-K) = Q(1+K:KT-K,1+K:KT-K) + eye(KT-2*K);
Q(1:KT-K,1+K:KT) = Q(1:KT-K,1+K:KT) - eye(KT-K);
Q(1+K:KT,1:KT-K) = Q(1+K:KT,1:KT-K) - eye(KT-K);
%}
upper_right = -1*[zeros(KT-K, K), eye(KT-K);
                  zeros(K, KT)];
Q = 2*(eye(KT) + upper_right);
Q(1:K, 1:K) = eye(K);
Q(end-K+1:end, end-K+1:end) = eye(K);

f = @(x) 0;
% The constraint function g does all the work of computing signed distances
% and their gradients
g = @(x) g_collisions(x, dsafe, [K,T], make_robot_poly, obstacles);
h = @(x) 0;

% TODO: create linear inequality constraints to enforce that the
% displacement is smaller than .2 (componentwise) over each timestep
%YOUR_CODE_HERE;
A_ineq =  eye(KT) + upper_right;
A_ineq = A_ineq(1:end-K, :);
A_ineq = [A_ineq;-1*A_ineq];%% include x1 - x2 and x2-x1
%YOUR_CODE_HERE;
b_ineq =  .2*ones(2*(KT-K), 1);

% TODO: create linear equality constraints to fix the beginning and end of the
% trajectory to the values in traj_init(:,1) and traj_init(:,end)

%YOUR_CODE_HERE;
A_eq = zeros(2*K,KT);
A_eq(1:K,1:K) = eye(K);
A_eq(K+1:2*K,KT-K+1:KT) = eye(K);
%YOUR_CODE_HERE;
b_eq = [traj_init(:,1); traj_init(:,end)];

cfg = struct();
cfg.callback = @(x,~) plot_traj(make_robot_poly, obstacles, reshape(x,size(traj_init)));
cfg.initial_trust_box_size=.1;
cfg.g_use_numerical = false;
cfg.min_approx_improve = 1e-2;

x = penalty_sqp(x0, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg);