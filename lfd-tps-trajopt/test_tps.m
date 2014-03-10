close all;

lambda = 1; % TPS regularization coefficient

% scene correspondences
X_s = [1 1; 1 0; 0 0; 0 1]';
X_s_new = [1 1; 1 0; -1 0; 0 1]';
[d,n] = size(X_s);

% gripper trajectory
T = 20;
X_g = [ 0.5 * ones(1,T); linspace(0,1,T) ];

K = tps_kernel(X_s);

N = null([X_s; ones(1,n)]);
sqrtKN = chol(N'*K*N);

cvx_begin
    variables Q(n-(d+1),d) B(d,d) c(d) s;

    minimize (pow_pos(norm(X_s_new' - K*N*Q - X_s'*B - ones(n,1)*c'),2) + lambda * sum(sum_square(sqrtKN*Q)))
    % equivalent to
    % minimize (norm(X_s_new' - K*A - X_s'*B - ones(n,1)*c')^2 + lambda * trace(A'*K*A))

    subject to
    ones(1, n)*(N*Q) == zeros(1, d);
cvx_end

orig_fig = figure();
warp_fig = figure();

warp = make_warp(N * Q, B, c, X_s);

figure(orig_fig);
hold on;
scatter(X_s(1,:), X_s(2,:), 'red');
scatter(X_g(1,:), X_g(2,:), 'cyan', 'x');
legend('x_i^{(S)}', 'x_t^{(G)}');

figure(warp_fig);
hold on;
X_s_warped = warp_pts(X_s, warp);
X_g_warped = warp_pts(X_g, warp);
scatter(X_s_new(1,:), X_s_new(2,:), 50, 'green');
scatter(X_s_warped(1,:), X_s_warped(2,:), 50, 'red');
scatter(X_g_warped(1,:), X_g_warped(2,:), 50, 'cyan', 'x');
draw_grid([0 1], [1 0], warp, 5, orig_fig, warp_fig);
legend('x_i^{(S)}\prime','f(x_i^{(S)})','f(x_t^{(G)})');


%{
% kind-of SCO formulation

num_params = n*d + d*d + d;

A_eq = zeros(2*d, num_params);
for i = 1:d
     A_eq(i, (i-1)*n+1:i*n) = X(:,i)';
     A_eq(i+d, (i-1)*n+1:i*n) = ones(1, n);
end
b_eq = zeros(2*d, 1);
A_ineq = zeros(1, num_params);
b_ineq = 0;
g = @(x) 0;
h = @(x) 0;
user_cfg = struct();
x0 = zeros(num_params, 1);
f = @(x) compute_f(x, X, Y, K);
Q = ones(num_params);
q = ones(num_params, 1)';
[x,success] = penalty_sqp(x0, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, user_cfg)
%}

%{
function val = compute_f(params, X, Y, K)
   [n, d] = size(X);
   params =  reshape(params, d, []);
   A = params(:, 1:n)';
   B = params(:, n+1:n+d)';
   c = params(:, end);
   temp = (Y - K*A - X*B - ones(n,1)*c');
   val = trace(temp' * temp) + trace(A' * K * A);
end
%}
