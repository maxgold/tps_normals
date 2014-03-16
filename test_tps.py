import numpy as np, numpy.linalg as nlg
from mayavi import mlab
import pylab

import tps_eval.tps_utils as tu



def disp_pts(points, normals, color1=(1,0,0), color2=(0,1,0)):
    """
    Ankush's plotting code.
    """
        
    figure = mlab.gcf()
    mlab.clf()
    figure.scene.disable_render = True

    points_glyphs   = mlab.points3d(points[:,0], points[:,1], points[:,2], color=color1, resolution=20, scale_factor=0.01)
    normals_glyphs   = mlab.points3d(normals[:,0], normals[:,1], normals[:,2], color=color2, resolution=20, scale_factor=0.01)
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
    mlab.show()

def test_normals ():
    """
    Test normals.
    """
    x = np.arange(100).astype('f')/10.0
    y = np.sin(x)
    xy = np.r_[np.atleast_2d(x),np.atleast_2d(y)].T
    nms = tu.find_all_normals_naive(xy,wsize=0.15)
    xy_nm = xy + nms*0.2

    z = x*0
    xyz = np.c_[xy,z]
    xyz_nm = np.c_[xy_nm,z]
    
    disp_pts(xyz, xyz_nm)

def test_normals2 ():
    """
    Test normals.
    """
    x = np.arange(100).astype('f')/10.0
    y = np.sin(x) + np.random.random((1,x.shape[0]))*0.05
    raw_input(y.shape)
    xy = np.r_[np.atleast_2d(x),np.atleast_2d(y)].T
    nms = tu.find_all_normals_naive(xy,wsize=0.15)
    xy_nm = xy + nms*0.2

    z = x*0
    xyz = np.c_[xy,z]
    xyz_nm = np.c_[xy_nm,z]
    
    disp_pts(xyz, xyz_nm)

def test_normals3 ():
    """
    Test normals.
    """
    angs = np.linspace(0, np.pi, 100)
    x = np.sin(angs) + np.random.random((1,angs.shape[0]))*0.05
    y = np.cos(angs) + np.random.random((1,angs.shape[0]))*0.05
    raw_input(y.shape)
    
    xy = np.r_[np.atleast_2d(x),np.atleast_2d(y)].T
    nms = tu.find_all_normals_naive(xy,wsize=0.15)
    xy_nm = xy + nms*0.03

    z = angs*0
    xyz = np.c_[xy,z]
    xyz_nm = np.c_[xy_nm,z]
    
    disp_pts(xyz, xyz_nm)


# x = np.arange(100).astype('f')/10.0
# y = np.sin(x)
# z = x*0
# xyz1 = np.r_[np.atleast_2d(x),np.atleast_2d(y),np.atleast_2d(z)].T
# xyz2 = np.r_[np.atleast_2d(x),np.atleast_2d(y),np.atleast_2d(z)].T + np.array([0.1,0.1,0.1])
