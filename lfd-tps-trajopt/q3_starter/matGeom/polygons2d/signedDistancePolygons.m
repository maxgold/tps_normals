function [dist, pts] = signedDistancePolygons(poly1, poly2)
%SIGNEDDISTANCEPOLYGONS Compute the signed distance between 2 polygons
%   DIST = signedDistancePolygons(POLY1, POLY2)
%   Returns the signed distance between 2 polygons
%
%   [DIST, POINTS] = signedDistancePolygons(POLY1, POLY2)
%   Also returns the 2 points involved with the distance. The
%   first point belongs to POLY1 and the second point belongs to POLY2.
%
%   Example
%   signedDistancePolygons
%
%   See also
%   distancePolygons, penetrationDepth
%
%
% ------
% Author: Alex Lee

% check if the polygons are intersecting each other
epsilon = 1e-9;
intersecting = false;

if any(isPointInPolygon(poly1, poly2)) || any(isPointInPolygon(poly2, poly1))
    intersecting = true;
end

% check every edge of poly1 for intersection with poly2
for i = 1:(size(poly1, 1) - 1)
    edge = [poly1(i,:), poly1(i+1,:)];
    if intersectEdgePolygon(edge, poly2)
        intersecting = true;
    end
end

% check edge between last vertex and first vertex
edge = [poly1(end,:), poly1(1,:)];
if intersectEdgePolygon(edge, poly2)
    intersecting = true;
end

if intersecting
    [dist, pts] = distancePolygons(poly1, poly2);
    
    % This check addresses the corner case where the two polygons' edges
    % are just touching but the polygons are not in collision - in this
    % case we want to use the points from distancePolygons
    if abs(dist) > epsilon
        [dist, pts] = penetrationDepth(poly1, poly2);
    end
else
    [dist, pts] = distancePolygons(poly1, poly2);
end
