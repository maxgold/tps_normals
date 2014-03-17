import numpy as np
from mayavi import mlab

import tn_eval.tps_utils as tu
from tn_utils import clouds
from tn_utils.colorize import colorize
from tn_eval.baseline import tps_rpm_bij_normals
from tn_rapprentice.registration import tps_rpm_bij
from tn_visualization import mayavi_utils
from tn_visualization.mayavi_plotter import PlotterInit, gen_custom_request, gen_mlab_request

def test_normals ():
    """
    Test normals.
    """
    x = np.arange(100).astype('f')/10.0
    y = np.sin(x)
    xy = np.r_[np.atleast_2d(x),np.atleast_2d(y)].T
    nms = tu.find_all_normals_naive(xy,wsize=0.15,flip_away=True)
    xy_nm = xy + nms*0.2

    z = x*0
    xyz = np.c_[xy,z]
    xyz_nm = np.c_[xy_nm,z]
    
    lines = [np.c_[p1,p2].T for p1,p2 in zip(xyz,xyz_nm)]
    
    mayavi_utils.disp_pts(xyz, xyz_nm)
    mayavi_utils.plot_lines(lines)
    mlab.show()

def test_normals2 ():
    """
    Test normals.
    """
    x = np.arange(100).astype('f')/10.0
    y = np.sin(x) + np.random.random((1,x.shape[0]))*0.05
    raw_input(y.shape)
    xy = np.r_[np.atleast_2d(x),np.atleast_2d(y)].T
    nms = tu.find_all_normals_naive(xy,wsize=0.15,flip_away=True)
    xy_nm = xy + nms*0.2

    z = x*0
    xyz = np.c_[xy,z]
    xyz_nm = np.c_[xy_nm,z]
    
    lines = [np.c_[p1,p2].T for p1,p2 in zip(xyz,xyz_nm)]
    
    mayavi_utils.disp_pts(xyz, xyz_nm)
    mayavi_utils.plot_lines(lines)
    mlab.show()

def test_normals3 ():
    """
    Test normals.
    """
    angs = np.linspace(0, np.pi, 100)
    x = np.sin(angs) + np.random.random((1,angs.shape[0]))*0.05
    y = np.cos(angs) + np.random.random((1,angs.shape[0]))*0.05

    
    xy = np.r_[np.atleast_2d(x),np.atleast_2d(y)].T
    nms = tu.find_all_normals_naive(xy,wsize=0.15,flip_away=True)
    xy_nm = xy + nms*0.03

    z = angs*0
    xyz = np.c_[xy,z]
    xyz_nm = np.c_[xy_nm,z]
    
    lines = [np.c_[p1,p2].T for p1,p2 in zip(xyz,xyz_nm)]
    
    mayavi_utils.disp_pts(xyz, xyz_nm)
    mayavi_utils.plot_lines(lines)
    mlab.show()

def test_normals_pts (pts, wsize = None, scale_factor=0.01):
    """
    Test normals.
    """
    if wsize is None: wsize = 0.02
    nms = tu.find_all_normals_naive(pts,wsize=wsize,flip_away=True,project_lower_dim=True)
    pts_nm = pts + nms*0.01
    
    lines = [np.c_[p1,p2].T for p1,p2 in zip(pts,pts_nm)]
    
    mayavi_utils.disp_pts(pts, pts_nm, scale_factor=scale_factor)
    mayavi_utils.plot_lines(lines)
    mlab.show()


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
        grid_lines = mayavi_utils.gen_grid2(f, mins=mins, maxes=maxes, xres=0.005, yres=0.005, zres=0.002)
    else:
        grid_lines = mayavi_utils.gen_grid(f, mins=mins, maxes=maxes)

    
    plotter_requests = []
    plotter_requests.append(gen_mlab_request(mlab.clf))
    plotter_requests.append(gen_custom_request('lines', lines=grid_lines, color=(0,0.5,0.3)))
    
    warped = f(src)
    
    plotter_requests.append(gen_mlab_request(mlab.points3d, src[:,0], src[:,1], src[:,2], color=(1,0,0), scale_factor=0.01))
    plotter_requests.append(gen_mlab_request(mlab.points3d, target[:,0], target[:,1], target[:,2], color=(0,0,1), scale_factor=0.01))
    plotter_requests.append(gen_mlab_request(mlab.points3d, warped[:,0], warped[:,1], warped[:,2], color=(0,1,0), scale_factor=0.01))

    if draw_plinks:
        plinks = [np.c_[ps, pw].T for ps,pw in zip(src, warped)]
        plotter_requests.append(gen_custom_request('lines', lines=plinks, color=(0.5,0,0), line_width=2, opacity=1))
                                
    return plotter_requests

def test_base_line (pts1, pts2):
    pts1 = clouds.downsample(pts1, 0.02)
    pts2 = clouds.downsample(pts2, 0.02)
    print pts1.shape
    print pts2.shape

    #plotter = PlotterInit()

    def plot_cb(src, targ, xtarg_nd, corr, wt_n, f):
        plot_requests = plot_warping(f.transform_points, src, targ, fine=False)
        for req in plot_requests:
            plotter.request(req)

    f1,_ = tps_rpm_bij(pts1, pts2, reg_init=10, reg_final=1, rot_reg=np.r_[1e-3,1e-3,1e-1], n_iter=50, plot_cb=plot_cb, plotting=0)
    #raw_input("Done with tps_rpm_bij")
    #plotter.request(gen_mlab_request(mlab.clf))
    f2,_ = tps_rpm_bij_normals(pts1, pts2, reg_init=10, reg_final=01, n_iter=50, rot_reg=np.r_[1e-3,1e-3,1e-1], 
                                nwsize=0.04, neps=0.02, plot_cb=plot_cb, plotting =0)
    #raw_input('abcd')

    from tn_rapprentice import tps
    #print tps.tps_cost(f1.lin_ag, f1.trans_g, f1.w_ng, pts1, pts2, 1)
    #print tps.tps_cost(f2.lin_ag, f2.trans_g, f2.w_ng, pts1, pts2, 1)
    #plotter.request(gen_mlab_request(mlab.clf))
    mlab.figure(1)
    mayavi_utils.plot_warping(f1, pts1, pts2, fine=False, draw_plinks=True)
    #mlab.show()
    mlab.figure(2)
    #mlab.clf()
    mayavi_utils.plot_warping(f2, pts1, pts2, fine=False, draw_plinks=True)
    mlab.show()
    
    

if __name__=='__main__':
    import h5py
    hdfh = h5py.File('/media/data_/human_demos_DATA/demos/overhand120/overhand120.h5','r')
    pts1 = np.asarray(hdfh['demo00022']['seg00']['cloud_xyz'])
    pts2 = np.asarray(hdfh['demo00081']['seg00']['cloud_xyz'])
    test_base_line(pts1, pts2)