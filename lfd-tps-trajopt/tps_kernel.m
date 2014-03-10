function K = tps_kernel(X)
    [d,n] = size(X);
    K = zeros(n);
    for i = 1:n
        for j = 1:n
            r = norm(X(:,i) - X(:,j));
            if r == 0
                K(i,j) = 0;
            else
                if d==2 || d==4
                    K(i,j) = (r^(4-d))*log(r);
                else
                    K(i,j) = -r^(4-d);
                end
            end
        end
    end
end
