function plot_path_2d(x,info)
%% PLOT_PATH_2D plots solution path and trust region
% Call with no arguments to clear out the stored solution
persistent h xs
if nargin == 0,
%     if exist('h','var') && ishandle(h), delete(h); end
    h = [];
    xs = [];
    return;
end
if ~isempty(h)
    delete(h); 
end
fprintf('x: %s\n',mat2str(x));

assert(all(size(x)==[2,1]));
xs = [xs,x];
h=plot(xs(1,:),xs(2,:),'kx-');
if isfield(info, 'trust_box_size')
    tbs = info.trust_box_size;
    h = [h, rectangle('Position',[x(1)-tbs/2,x(2)-tbs/2,tbs,tbs])];
end            
pause(.01);
end