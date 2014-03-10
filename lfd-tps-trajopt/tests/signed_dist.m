obstacle = orientedBoxToPolygon([1.5 0 1 0.5 0]);
link = orientedBoxToPolygon([1.5 0.1 2 0.2 0]);

[d,pts] = signedDistancePolygons(link, obstacle);

d
figure;
plot(pts(:,1), pts(:,2), 'xr');
hold on;
drawPolygon(obstacle);
drawPolygon(link);