function h = plot_normals(Xe, Xn, plot_opt)
    lines = [Xe; Xe+Xn; NaN(size(Xe))];
    lines = reshape(reshape(lines,[],1),2,[]);
    h = plot(lines(1,:), lines(2,:), plot_opt);
end
