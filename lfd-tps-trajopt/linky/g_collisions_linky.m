function [val,jac] = g_collisions_linky(l, x, dsafe, traj_shape, make_robot_poly, obstacles)
assert(size(x,2)==1);
traj = reshape(x, traj_shape);
[K,T] = size(traj);
L = length(l);
assert(K == L);

val = zeros(L*length(obstacles)*T,1);
jac = zeros(size(val,1), size(x,1));

icontact = 1;


for t=1:T
    xt = traj(:,t);
    for iobs=1:length(obstacles)
        robot_polys = make_robot_poly(xt);
        for ipoly=1:L
            [d,pts] = signedDistancePolygons(...
                    robot_polys{ipoly}, ...
                    obstacles{iobs});
            ptOnRobot = pts(1,:);
            ptOnObs = pts(2,:);
            normalObsToRobot = -sign(d)*normalize(ptOnRobot - ptOnObs);

            gradd = normalObsToRobot * [calc_jac_linky(ptOnRobot, l(1:ipoly), xt(1:ipoly)) zeros(2, L-ipoly)];

            val(icontact) = dsafe - d;
            jac(icontact,K*(t-1)+1:K*t) = gradd;
            icontact = icontact+1;
        end
    end
end

end