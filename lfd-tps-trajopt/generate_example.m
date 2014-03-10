function[X, Y, demo_obs, test_obs, demo_traj] = generate_example(demo_dim, test_dim)
    close all
    draw_borders = [demo_dim(1), demo_dim(3), demo_dim(2), demo_dim(4)];
    axis manual
    hold on
    f1 = figure(1);
    axis(demo_dim);
    f2 = figure(2);
    axis(test_dim)
    disp('Create demo obstacles, by clicking in window. Press enter when done')
    stop = false;
    demo_obs = {};
    while ~stop
        demo_obs = add_obstacle(demo_obs, 1, demo_dim);
        resp = input('add another figure?[y/n]', 's');
        stop = (resp == 'n');
    end
    test_obs = {};
    stop = false;
    disp('Create test obstacles, by clicking in window. Press enter when done')
    while ~stop
        test_obs = add_obstacle(test_obs, 2, test_dim);
        resp = input('add another figure?[y/n]', 's');
        stop = (resp == 'n'); 
    end
    X = [];
    Y = [];
    num_pts = input('Enter the number of corresponding points you would like to include');
    while num_pts > 0
        [x, y] = get_corresponding_pt(demo_dim, test_dim);
        X = [X;x];
        Y = [Y;y];      
        num_pts = num_pts - 1;
	if size(X, 1) >= 5
	  plot_current_warp(X', Y', demo_obs, test_obs, draw_borders);
	end
        if num_pts == 0
            num_pts = input('Total number of points entered. \nIf you would like to enter more, do so now, otherwise enter 0');
        end
    end
    X = X';
    Y = Y';
    demo_traj = input_demo_traj(X, Y, demo_obs, test_obs, draw_borders);
end

function plot_current_warp(X, Y, demo_obs, test_obs, dim)
  [~, warp_fn, ~] = compute_warp(X, Y);
  clf;
  demo_fig = figure(1);
  test_fig = figure(2);
  upper_left = dim(1:2);
  bottom_right = dim(3:4);
  draw_grid(upper_left, bottom_right, warp_fn, 10, demo_fig, test_fig);
  figure(1);
  hold on;
  scatter(X(:,1), X(:,2));
  for obs_cell = demo_obs
    obs = obs_cell{1};
    plot([obs(:,1)',obs(1,1)],[obs(:,2)',obs(1,2)]);
  end  
  figure(2);
  hold on;
  scatter(Y(:,1), Y(:,2));
  for obs_cell = test_obs
    obs = obs_cell{1};
    plot([obs(:,1)',obs(1,1)],[obs(:,2)',obs(1,2)]);
  end  
end

function [p1, p2] = get_corresponding_pt(dim1, dim2)
    disp('Select a point in the demo scene')
    figure(1);
    p1 = ginput(1);
    scatter(p1(1), p1(2));
    axis(dim1)
    disp('Select a corresponding point in the test scene')
    figure(2);
    p2 = ginput(1);
    scatter(p2(1), p2(2));
    axis(dim2)
end

function obstacles = add_obstacle(obstacles, fig, dim)
figure(fig);
hold on;
axis(dim);
for obs_cell = obstacles
    obs = obs_cell{1};
    plot([obs(:,1)',obs(1,1)],[obs(:,2)',obs(1,2)]);
end
[x,y] = ginput;
% c_indices = convhull(x, y);
% x = x(c_indices);
% y = y(c_indices);
obstacles{end+1} = [x,y];
plot([x',x(1)], [y',y(1)]);
end