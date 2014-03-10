close all;
clear;

% scene correspondences
X_s = [1 1; 1 0; 0 0; 0 1]';
X_s_new = [1 1; 1 0; -1 0; 0 1]';
Xe_s = [1 1; 1 0; 0 0; 1 1; 1 0; 0 0]';
Xn_s = [0 1; 0 1; 0 1; 1 0; 1 0; 1 0]';
Xn_s_new = [0 1; 0 1; 0 1; 1 0; 1 0; 1 0]';
% Xe_s = [.5 .5; .5 .5]';
% Xn_s = [0 1; 1 0]';
% Xn_s_new = [0 1; 1 0]';
% Xe_s = [.75 .75; .75 .25; .25 .25; .25 .75]';
% Xn_s = [0 1; 0 1; 0 1; 0 1]';
% Xn_s_new = [0 1; 0 1; 0 1; 0 1]';
[dim,k] = size(X_s);
assert(all(size(X_s) == size(X_s_new)));
[~,m] = size(Xe_s);
assert(size(Xe_s,1) == dim);
assert(all(size(Xe_s) == size(Xn_s)));
assert(all(size(Xe_s) == size(Xn_s_new)));

delta = 0.1;

L = [tps_kernel(X_s) ones(k,1) X_s'; ones(1,k) zeros(1,1+dim); X_s zeros(dim,1+dim)];
d = [X_s_new'; zeros(dim+1,dim)];
w = L\d;

P = zeros(m);
for i = 1:m
    for j = 1:m
        P(i,j) = der2_U(Xe_s(:,i) - Xe_s(:,j), Xn_s(:,j), Xn_s(:,i)); 
    end
end

M = zeros(k+3,m);
for i = 1:k
    for j = 1:m
        M(i,j) = der_U(Xe_s(:,j) - X_s(:,i), Xn_s(:,j));
    end
end
M(end-1:end,:) = Xn_s;

N = P + M'*(L\M);

warp_orig = make_warp_normal(w, zeros(0), X_s, zeros(0), zeros(0));
Xn_s_warped_orig = warp_normals(Xe_s, Xn_s, warp_orig);

%{
Q = -2*log(delta)*(eye(2*m) + (1/(2*log(delta)))*[N zeros(size(N)); zeros(size(N)) N]);
d_til = Xn_s_new' - Xn_s_warped_orig';
d_til = reshape(d_til,[],1);
w_til = Q\d_til;
w_til = reshape(w_til,[],2);
%}

Q = -2*log(delta)*(eye(m) + (1/(2*log(delta)))*N);
d_til = Xn_s_new' - Xn_s_warped_orig';
w_til = Q\d_til;

warp = make_warp_normal(w, w_til, X_s, Xe_s, Xn_s);

grid_n_seg = 10;

orig_fig = figure();
figure(orig_fig);
hold on;
draw_grid_orig([0 1], [1 0], grid_n_seg, 'b-');
X_s_h = scatter(X_s(1,:), X_s(2,:), 'r');
Xn_s_h = plot_normals(Xe_s, delta*Xn_s, 'r');
legend([X_s_h Xn_s_h], 'x', 'x_n');

warp_fig = figure();
figure(warp_fig);
hold on;
X_s_new_h = scatter(X_s_new(1,:), X_s_new(2,:), 50, 'g');
% plot landmark-only warp
draw_grid_warp([0 1], [1 0], warp_orig, grid_n_seg, 'c:');
X_s_warped_orig = warp_pts(X_s, warp_orig);
X_s_warped_orig_h = scatter(X_s_warped_orig(1,:), X_s_warped_orig(2,:), 50, 'm');
Xe_s_warped_orig = warp_pts(Xe_s, warp_orig);
Xn_s_warped_orig = warp_normals(Xe_s, Xn_s, warp_orig);
Xn_s_warped_orig_h = plot_normals(Xe_s_warped_orig, delta*Xn_s_warped_orig, 'm');
Xn_s_new_warped_orig_h = plot_normals(Xe_s_warped_orig, delta*Xn_s_new, 'g');
% plot normal-constrained warp
draw_grid_warp([0 1], [1 0], warp, grid_n_seg, 'b-');
X_s_warped = warp_pts(X_s, warp);
X_s_warped_h = scatter(X_s_warped(1,:), X_s_warped(2,:), 50, 'r');
Xe_s_warped = warp_pts(Xe_s, warp);
Xn_s_warped = warp_normals(Xe_s, Xn_s, warp);
Xn_s_warped_h = plot_normals(Xe_s_warped, delta*Xn_s_warped, 'r');
Xn_s_new_warped_h = plot_normals(Xe_s_warped, delta*Xn_s_new, 'g');
legend([X_s_new_h Xn_s_new_warped_orig_h Xn_s_new_warped_h ...
    X_s_warped_orig_h Xn_s_warped_orig_h ...
    X_s_warped_h Xn_s_warped_h ],   ...
    'x^\prime', 'x_n^\prime', 'x_n^\prime', ...
    'f_{landmark-only}(x)', 'f_{landmark-only}(x_n)', ...
    'f_{normals}(x)', 'f_{normals}(x_n)');
