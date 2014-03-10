function warped = warp_normals(pts, normals, warp)
    warped = zeros(size(normals));
    for i=1:size(normals, 2)
        warped(:,i) = numerical_jac(warp, pts(:,i)) * normals(:,i);
    end
end
