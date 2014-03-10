function [ X_s, X_s_new, X_g, obstacles, obstacles_new ] = generate_tunnel_ex()

obstacles = {orientedBoxToPolygon([2 0.5 1 0.5 0]), ...
    orientedBoxToPolygon([3.5 0.75 1 0.5 0]), ...
    orientedBoxToPolygon([5 0.5 1 0.5 0]), ...
    orientedBoxToPolygon([2 -0.5 1 0.5 0]), ...
    orientedBoxToPolygon([3.5 -0.25 1 0.5 0]), ...
    orientedBoxToPolygon([5 -0.5 1 0.5 0])};

T1 = 15;
T2 = 5;
T3 = 15;
T4 = 5;
T5 = 15;
traj_link1 = [linspace(1, 2.75, T1); zeros(4,T1)];
traj_link2 = [2.75 * ones(1,T2); linspace(0, 0.25, T2); zeros(3,T2)];
traj_link3 = [[2.75; 0.25] * ones(1,T3); linspace(0, 1.5, T3); zeros(2,T3)];
traj_link4 = [[2.75; 0.25; 1.5] * ones(1,T4); linspace(0, 0.25, T4); zeros(1,T4)];
traj_link5 = [[2.75; 0.25; 1.5; 0.25] * ones(1,T5); linspace(0, 1.75, T5);];

X_g = [traj_link1, traj_link2, traj_link3, traj_link4, traj_link5];

corr_pts = zeros(2, length(obstacles)*4 + 2);
for i=1:length(obstacles)
    corr_pts(:,(i-1)*4 + 1:i*4) = obstacles{i}';
end
corr_pts(:,end-1) = [1;0];
corr_pts(:,end) = [6;0];
X_s = corr_pts;

new_obstacles = {orientedBoxToPolygon([3.5 1.25 1 0.5 0]), ...
    orientedBoxToPolygon([3.5 0.25 1 0.5 0])};
new_corr_pts = corr_pts;
new_corr_pts(:,(2-1)*4 + 1:2*4) = new_obstacles{1}';
new_corr_pts(:,(5-1)*4 + 1:5*4) = new_obstacles{2}';
X_s_new = new_corr_pts;

obstacles_new = obstacles;
obstacles_new{2} = new_obstacles{1};
obstacles_new{5} = new_obstacles{2};

end

