function [y, grad, hess] = kinematic_cost_linky(l, x, traj_shape, ts, target_pts)
    [y, grad] = kinematic_cnt_linky(l, x, traj_shape, ts, target_pts);
    hess = 2*(grad'*grad);
    grad = 2*(y'*grad);
    y = y'*y;
    hess = (hess + hess')/2;
    % diagonal adjustment
    mineig = min(eig(hess));
    if mineig < 0
        fprintf('    negative hessian detected. adjusting by %.3g\n',-mineig);
        hess = hess + eye(size(hess,1)) * ( - mineig);
    end
end
