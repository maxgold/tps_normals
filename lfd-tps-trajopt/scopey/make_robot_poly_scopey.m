function robot_polys = make_robot_poly_scopey( angles, robot_width, x )
    assert(size(x, 2)==1);
    assert(length(angles) == length(x));
    x_dim = length(x);
    
    ang_abs = 0;
    x_offset = 0;
    y_offset = 0;
    robot_polys = cell(1,x_dim);
    for i = 1:x_dim
        ang_abs = ang_abs + angles(i);
        x_offset_local = cos(ang_abs)*x(i)/2;
        y_offset_local = sin(ang_abs)*x(i)/2;
        robot_polys{1,i} = orientedBoxToPolygon([x_offset+x_offset_local, y_offset+y_offset_local, x(i), robot_width, rad2deg(ang_abs)]);
        x_offset = x_offset + 2*x_offset_local;
        y_offset = y_offset + 2*y_offset_local;
    end
end