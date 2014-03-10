axis_range = [-0.5 6.5 -2 2];

clf;
hold on;
axis equal;
axis(axis_range);

robot_angles = deg2rad([0;90;-90;-90;90]);
robot_width = 0.2;
make_robot_poly = @(x) make_robot_poly_scopey(robot_angles, robot_width, x);

x = [2.5 1 1.5 2 1]';
robot_poly = make_robot_poly(x);

drawPolygon(robot_poly, 'k','LineWidth',2);
