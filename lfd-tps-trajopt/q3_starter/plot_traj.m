function plot_traj(make_robot_poly, obstacles, traj, varargin)

animate = true;
axis_range = [-21 21 -21 21]/10;
for opt = varargin
    if islogical(opt{:})
        animate = opt{:};
    else
        axis_range = opt{:};
    end
end

clf;
hold on;
axis equal;
axis(axis_range);

for obstacle = obstacles
    drawPolygon(obstacle{:},'b');
end



h2 = [];
for t=1:size(traj,2)
    x = traj(:,t);
    robot_polys = make_robot_poly(x);
    if ~iscell(robot_polys)
        robot_polys = {robot_polys};
    end
    c_param = (t-1)/(size(traj,2));
    for robot_poly = robot_polys
        h2(end+1) = drawPolygon(robot_poly{:}, 'Color', [1-c_param, c_param, 0]);
    end
end


if animate
%     set(gcf,'color','w');
%     set(gca,'Visible','off');
%     writerObj = VideoWriter('demo/images/two_step_traj.avi');
%     writerObj.FrameRate = 10;
%     open(writerObj);
    for t=1:size(traj,2)
        h1=[];
        x = traj(:,t);
        robot_polys = make_robot_poly(x);
        if ~iscell(robot_polys)
            robot_polys = {robot_polys};
        end
        for robot_poly = robot_polys
            h1(end+1) = drawPolygon(robot_poly{:}, 'k','LineWidth',2);
            % draw distances to obstacles
            %{
            for obstacle = obstacles
                [dist, contact_pts] = signedDistancePolygons(robot_poly{:}, obstacle{:});
                edge = createEdge(contact_pts(1,:), contact_pts(2,:));
                if dist < 0, color='r'; else color='g'; end;
                h1(end+1) = drawEdge(edge, color, 'LineWidth', 3);
            end
            %}
        end
%         frame = getframe;
%         writeVideo(writerObj,frame);
        pause(.04);
        delete(h1);
    end
%     close(writerObj);
end

end
