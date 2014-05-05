import numpy as np
from mayavi import mlab
from tn_utils.colorize import colorize

def disp_pts(points, normals, color1=(1,0,0), color2=(0,1,0), scale_factor=0.01):
    """
    Ankush's plotting code.
    """
        
    figure = mlab.gcf()
    mlab.clf()
    figure.scene.disable_render = True

    points_glyphs   = mlab.points3d(points[:,0], points[:,1], points[:,2], color=color1, resolution=20, scale_factor=scale_factor)
    normals_glyphs   = mlab.points3d(normals[:,0], normals[:,1], normals[:,2], color=color2, resolution=20, scale_factor=scale_factor)
    glyph_points1 = points_glyphs.glyph.glyph_source.glyph_source.output.points.to_array()
    glyph_points2 = normals_glyphs.glyph.glyph_source.glyph_source.output.points.to_array()

    dd = 0.001

    outline1 = mlab.outline(points_glyphs, line_width=3)
    outline1.outline_mode = 'full'
    p1x, p1y, p1z = points[0,:]
    outline1.bounds = (p1x-dd, p1x+dd,
                       p1y-dd, p1y+dd,
                       p1z-dd, p1z+dd)

    pt_id1 = mlab.text(0.8, 0.2, '0 .', width=0.1, color=color1)

    outline2 = mlab.outline(normals_glyphs, line_width=3)
    outline2.outline_mode = 'full'
    p2x, p2y, p2z = normals[0,:]
    outline2.bounds = (p2x-dd, p2x+dd,
                       p2y-dd, p2y+dd,
                       p2z-dd, p2z+dd)  
    pt_id2 = mlab.text(0.8, 0.01, '0 .', width=0.1, color=color2)
    
    figure.scene.disable_render = False


    def picker_callback(picker):
        """ Picker callback: this gets called during pick events.
        """
        if picker.actor in points_glyphs.actor.actors:
            point_id = picker.point_id/glyph_points1.shape[0]
            if point_id != -1:
                ### show the point id
                pt_id1.text = '%d .'%point_id
                #mlab.title('%d'%point_id)
                x, y, z = points[point_id,:]
                outline1.bounds = (x-dd, x+dd,
                                   y-dd, y+dd,
                                   z-dd, z+dd)
        elif picker.actor in normals_glyphs.actor.actors:
            point_id = picker.point_id/glyph_points2.shape[0]
            if point_id != -1:
                ### show the point id
                pt_id2.text = '%d .'%point_id
                x, y, z = normals[point_id,:]
                outline2.bounds = (x-dd, x+dd,
                                   y-dd, y+dd,
                                   z-dd, z+dd)


    picker = figure.on_mouse_pick(picker_callback)
    picker.tolerance = dd/2.

def gen_grid(f, mins, maxes, ncoarse=10, nfine=30):
    """
    generate 3d grid and warps it using the function f.
    The grid is based on the number of lines (ncoarse & nfine).
    """
    dim = len(mins)
    if dim ==3:
        xmin, ymin, zmin = mins
        xmax, ymax, zmax = maxes
    elif dim==2:
        xmin, ymin = mins
        xmax, ymax = maxes
    else: raise NotImplemented()

    xcoarse = np.linspace(xmin, xmax, ncoarse)
    ycoarse = np.linspace(ymin, ymax, ncoarse)
    if dim == 3:
        zcoarse = np.linspace(zmin, zmax, ncoarse)

    xfine = np.linspace(xmin, xmax, nfine)
    yfine = np.linspace(ymin, ymax, nfine)
    if dim == 3:
        zfine = np.linspace(zmin, zmax, nfine)
    
    lines = []
    if dim == 3:
        if len(zcoarse) > 1:
            for x in xcoarse:
                for y in ycoarse:
                    xyz = np.zeros((nfine, dim))
                    xyz[:,0] = x
                    xyz[:,1] = y
                    xyz[:,2] = zfine
                    lines.append(f(xyz))
    
        for y in ycoarse:
            for z in zcoarse:
                xyz = np.zeros((nfine, dim))
                xyz[:,0] = xfine
                xyz[:,1] = y
                xyz[:,2] = z
                lines.append(f(xyz))
            
        for z in zcoarse:
            for x in xcoarse:
                xyz = np.zeros((nfine, 3))
                xyz[:,0] = x
                xyz[:,1] = yfine
                xyz[:,2] = z
                lines.append(f(xyz))
    else: 
        for y in ycoarse:
            xyz = np.zeros((nfine, dim))
            xyz[:,0] = xfine
            xyz[:,1] = y
            lines.append(f(xyz))
            
        for x in xcoarse:
            xyz = np.zeros((nfine, dim))
            xyz[:,0] = x
            xyz[:,1] = yfine
            lines.append(f(xyz))

    return lines


def gen_grid2(f, mins, maxes, xres = .01, yres = .01, zres = .01):
    """
    generate 3d grid and warps it using the function f.
    The grid is based on the resolution specified.
    """
    dim = len(mins)
    if dim ==3:
        xmin, ymin, zmin = mins
        xmax, ymax, zmax = maxes
    elif dim==2:
        xmin, ymin = mins
        xmax, ymax = maxes
    else: raise NotImplemented()

    xcoarse = np.arange(xmin, xmax+xres/10., xres)
    ycoarse = np.arange(ymin, ymax+yres/10., yres)
    if dim == 3:
        zcoarse = np.arange(zmin, zmax+zres/10., zres)

    xfine = np.arange(xmin, xmax+xres/10., xres/5.)
    yfine = np.arange(ymin, ymax+yres/10., yres/5.)
    if dim == 3:
        zfine = np.arange(zmin, zmax+zres/10., zres/5.)

    lines = []
    if dim == 3:
        if len(zcoarse) > 1:
            for x in xcoarse:
                for y in ycoarse:
                    xyz = np.zeros((len(zfine), 3))
                    xyz[:,0] = x
                    xyz[:,1] = y
                    xyz[:,2] = zfine
                    lines.append(f(xyz))
    
        for y in ycoarse:
            for z in zcoarse:
                xyz = np.zeros((len(xfine), 3))
                xyz[:,0] = xfine
                xyz[:,1] = y
                xyz[:,2] = z
                lines.append(f(xyz))
            
        for z in zcoarse:
            for x in xcoarse:
                xyz = np.zeros((len(yfine), 3))
                xyz[:,0] = x
                xyz[:,1] = yfine
                xyz[:,2] = z
                lines.append(f(xyz))
    else:
        for y in ycoarse:
            xyz = np.zeros((len(xfine), dim))
            xyz[:,0] = xfine
            xyz[:,1] = y
            lines.append(f(xyz))
            
        for x in xcoarse:
            xyz = np.zeros((len(yfine), dim))
            xyz[:,0] = x
            xyz[:,1] = yfine
            lines.append(f(xyz))

    return lines

def plot_lines(lines, color=(1,1,1), line_width=1, opacity=0.4):
    """
    input  :
    
      - lines :  a LIST of m matrices of shape n_ix3
                 each matrix is interpreted as one line
      - color : (r,g,b) values for the lines
      - line_width : width of the lines
      - opacity    : opacity of the lines


    output : plot each line in mayavi
    
    adapted from : http://docs.enthought.com/mayavi/mayavi/auto/example_plotting_many_lines.html
    
    call
    mlab.show() to actually display the grid, after this function returns
    """

    Ns   = np.cumsum(np.array([l.shape[0] for l in lines]))
    Ntot = Ns[-1]
    Ns   = Ns[:-1]-1
    connects  = np.vstack([np.arange(0, Ntot-1.5), np.arange(1,Ntot-0.5)]).T
    connects  = np.delete(connects, Ns, axis=0)
    
    pts = np.vstack(lines)
    dim = pts.shape[1]
    if dim == 2:
        pts = np.c_[pts,np.zeros((pts.shape[0],1))]
    s   = np.ones(pts.shape[0])

    # Create the points
    
    src = mlab.pipeline.scalar_scatter(pts[:,0], pts[:,1], pts[:,2], s)
    src.mlab_source.dataset.lines = connects
    lines = mlab.pipeline.stripper(src)

    # Finally, display the set of lines
    surf = mlab.pipeline.surface(lines, line_width=line_width, opacity=opacity)

    # set the color of the lines
    r,g,b = color
    color = 255*np.array((r,g,b, 1))
    surf.module_manager.scalar_lut_manager.lut.table = np.array([color, color])
    
    
def plot_transform(T, size=0.1):
    """
    plots the transform represented by
    the 4x4 transformation matrix T.
    """
    assert T.shape==(4,4)
    origin     = np.c_[T[0:3,3]]
    origin_mat = np.repeat(origin, 3, axis=1).T
    mlab.quiver3d(np.c_[origin[0]], np.c_[origin[1]], np.c_[origin[2]],
                  np.c_[T[0,0]], np.c_[T[1,0]], np.c_[T[2,0]], color=(1,0,0), line_width=3, scale_factor=size)
    mlab.quiver3d(np.c_[origin[0]], np.c_[origin[1]], np.c_[origin[2]],
                  np.c_[T[0,1]], np.c_[T[1,1]], np.c_[T[2,1]], color=(0,1,0), line_width=3, scale_factor=size)
    mlab.quiver3d(np.c_[origin[0]], np.c_[origin[1]], np.c_[origin[2]],
                  np.c_[T[0,2]], np.c_[T[1,2]], np.c_[T[2,2]], color=(0,0,1), line_width=3, scale_factor=size)


def plot_warping(f, src, target, fine=True, draw_plinks=True):
    """
    function to plot the warping as defined by the function f.
    src : nx3 array
    target : nx3 array
    fine : if fine grid else coarse grid.
    """
    print colorize("Plotting grid ...", 'blue', True)
    mean = np.mean(src, axis=0)

    print '\tmean : ', mean
    print '\tmins : ', np.min(src, axis=0)
    print '\tmaxes : ', np.max(src, axis=0)

    mins = np.min(src, axis=0)#mean + [-0.1, -0.1, -0.01]
    maxes = np.max(src, axis=0)#mean + [0.1, 0.1, 0.01]


    grid_lines = []
    if fine:
        grid_lines = gen_grid2(f.transform_points, mins=mins, maxes=maxes, xres=0.005, yres=0.005, zres=0.002)
    else:
        grid_lines = gen_grid(f.transform_points, mins=mins, maxes=maxes)

    
    plot_lines(grid_lines, color=(0,0.5,0.3))
    
    warped = f.transform_points(src)
    if src.shape[1] == 2:
        src = np.c_[src,np.zeros((src.shape[0],1))]
        target = np.c_[target,np.zeros((target.shape[0],1))]
        warped = np.c_[warped,np.zeros((warped.shape[0],1))]
    
    mlab.points3d (src[:,0], src[:,1], src[:,2], color=(1,0,0), scale_factor=0.01)
    mlab.points3d (target[:,0], target[:,1], target[:,2], color=(0,0,1), scale_factor=0.01)
    mlab.points3d (warped[:,0], warped[:,1], warped[:,2], color=(0,1,0), scale_factor=0.01)

    if draw_plinks:
        plinks = [np.c_[ps, pw].T for ps,pw in zip(src, warped)]
        plot_lines (lines=plinks, color=(0.5,0,0), line_width=2, opacity=1)
