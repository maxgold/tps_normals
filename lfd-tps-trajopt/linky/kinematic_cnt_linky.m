function [y, grad] = kinematic_cnt_linky(l, x, traj_shape, ts, target_pts)
% ts = vector of time steps
% target_pts = #dim x #time steps

    fwd_kin_robot = @(x) fwd_kin_linky(l, x);
    calc_jac_robot = @(pt, x) calc_jac_linky(pt, l, x);
    
    [y, grad] = kinematic_cnt(x, traj_shape, ts, target_pts, fwd_kin_robot, calc_jac_robot);
end
