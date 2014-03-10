function [ traj_pts ] = get_traj_pts_scopey( getTraj, robot_angles, x )

    traj = getTraj(x);
    [~, T] = size(traj);
    traj_pts = zeros(2,T);
    if ~isnumeric(x)
        traj_pts = sym(traj_pts);
    end
    for i=1:T
        traj_pts(:,i) = fwd_kin_scopey(robot_angles, traj(:,i));
    end

end

