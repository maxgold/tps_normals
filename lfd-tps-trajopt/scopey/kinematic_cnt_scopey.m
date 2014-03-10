function [y, grad] = kinematic_cnt_scopey(angles, x, traj_shape, ts, target_pts)
% ts = vector of time steps
% target_pts = #dim x #time steps

    fwd_kin_robot = @(x) fwd_kin_scopey(angles, x);
    calc_jac_robot = @(pt, x) calc_jac_scopey(pt, angles, x);
    
    [y, grad] = kinematic_cnt(x, traj_shape, ts, target_pts, fwd_kin_robot, calc_jac_robot);
end

