function [val, jac] = g_collisions_scopey(angles, x, dsafe, traj_shape, make_robot_poly, obstacles)
% x should be of the form [trajectory,other_stuff] where traj_shape
% reshapes the trajectory
%
% Note: The returned "jac" is not always equal to the numerical jacobian -
% we suggest you use the numerical jacobian.

assert(size(x,2)==1);

if isempty(obstacles)
    val = 0;
    jac = zeros(1, size(x,1));
    return;
end

traj = reshape(x(1:traj_shape(1)*traj_shape(2)), traj_shape);
[K,T] = size(traj);
num_links = length(angles);
assert(K == num_links);
%%

val = zeros(num_links*length(obstacles)*T,1);
jac = zeros(size(val,1), size(x,1));

icontact = 1;

extdir = zeros(2,num_links);
id = eye(num_links);
for ipoly=1:num_links
    new_xt = id(:,ipoly);
    extdir(:,ipoly) = fwd_kin_scopey(angles(1:ipoly), new_xt(1:ipoly)) - ...
        fwd_kin_scopey(angles(1:ipoly), zeros(ipoly,1));
end

for t=1:T
    xt = traj(:,t);
    for iobs=1:length(obstacles)
        robot_polys = make_robot_poly(xt);
        for ipoly=1:num_links
            [d,pts] = signedDistancePolygons(...
                    robot_polys{ipoly}, ...
                    obstacles{iobs});
            ptOnRobot = pts(1,:);
            ptOnObs = pts(2,:);
            normalObsToRobot = -sign(d)*normalize(ptOnRobot - ptOnObs);

            gradd = normalObsToRobot * [calc_jac_scopey(ptOnRobot, angles(1:ipoly), xt(1:ipoly)) zeros(2, num_links-ipoly)];
            % Ignore gradient coords for links whose direction of movement
            % is an acute angle from the vector (ptOnRobot - ptOnObs)
            gradd(ipoly) = ((ptOnRobot - ptOnObs)*extdir(:, ipoly) < 0) * gradd(ipoly);

            val(icontact) = dsafe - d;
            jac(icontact,num_links*(t-1)+1:num_links*t) = gradd;
            icontact = icontact+1;
        end
    end
end
end

