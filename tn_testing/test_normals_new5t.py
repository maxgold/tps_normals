import numpy as np
from mayavi import mlab

import tn_eval.tps_utils as tu
from tn_utils import clouds
from tn_utils.colorize import colorize
from tn_eval.baseline import tps_rpm_bij_normals_naive, tps_rpm_bij_normals, fit_ThinPlateSpline_normals
from tn_eval import tps_evaluate as te
from tn_rapprentice.registration import tps_rpm_bij, fit_ThinPlateSpline
from tn_visualization import mayavi_utils
from tn_visualization.mayavi_plotter import PlotterInit, gen_custom_request, gen_mlab_request

def test_normals_pts (pts, nms = None, wsize = None, delta=0.01, scale_factor=0.01,show=False):
    """
    Test normals.
    """
    if wsize is None: wsize = 0.02
    if nms is None:
        nms = tu.find_all_normals_naive(pts,wsize=wsize,flip_away=True,project_lower_dim=True)
    pts_nm = pts + nms*delta
    
    lines = [np.c_[p1,p2].T for p1,p2 in zip(pts,pts_nm)]
    
    if show:
        mayavi_utils.disp_pts(pts, pts_nm, scale_factor=scale_factor)
    mayavi_utils.plot_lines(lines)
    if show:
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



def create_flap_points_normals(n, l, dim):
    if dim == 3:
        pts1 = np.r_[np.c_[np.zeros((n,1)), np.linspace(0,l,n), np.zeros((n,1))], np.c_[np.ones((n,1)), np.linspace(0,l,n), np.zeros((n,1))]]
        pts1[-1,-1] += l/4.0
        pts2 = pts1.copy()
        pts2[n:,-1] += l
        e1 = np.r_[np.tile([1.,0.,0.], (n,1)), np.tile([-1.,0.,0.], (n,1))]
        e2 = e1.copy()
    elif dim == 2:
        pts1 = np.r_[np.c_[np.zeros((n,1)), np.linspace(0,l,n)], np.c_[np.ones((n,1)), np.linspace(0,l,n)]]
        pts2 = pts1.copy()
        pts2[n:,-1] += 2*l
        e1 = np.r_[np.tile([1.,0.], (n,1)), np.tile([-1.,0.], (n,1))]
        e2 = e1.copy()
    
    return pts1, pts2, e1, e2

def test_normals_new5 ():

#pts1 = clouds.downsample(pts1, 0.02).astype('float64')
    
#     pts1 = np.array([[0.,0.,0.], [0.,0.5,0.], [0.,1.,0.], [1.,0.,0.0], [1.,0.5,0.0], [1.,1.,0.25]])
#     pts2 = np.array([[0.,0.,0.], [0.,0.5,0.], [0.,1.,0.], [1.,0.,1.], [1.,0.5,1.], [1.,1.,1.25]])
#     e1 = np.array([[1.,0.,0.], [1.,0.,0.], [1.,0.,0.], [-1.,0.,0.], [-1.,0.,0.], [-1.,0.,0.]])
#     e2 = np.array([[1.,0.,0.], [1.,0.,0.], [1.,0.,0.], [-1.,0.,0.], [-1.,0.,0.], [-1.,0.,0.]])
    
    pts1, pts2, e1, e2 = create_flap_points_normals(3.0,1,dim=3)
    f1 = fit_ThinPlateSpline(pts1, pts2, bend_coef=0.1, rot_coef=1e-5, wt_n=None, use_cvx=True)
    #f2 = fit_ThinPlateSpline(pts1, pts2, bend_coef=0.1, rot_coef=1e-5, wt_n=None, use_cvx=True)
    f2 = te.tps_eval(pts1, pts2, e1, e2, bend_coef=0.01, rot_coef=1e-5, wt_n=None, nwsize=0.15, delta=0.0001)
    #f2 = te.tps_fit_normals_cvx(pts1, pts2, e1, e2, bend_coef=0.1, rot_coef=1e-5, normal_coef=0.1, wt_n=None, nwsize=0.15, delta=0.0001)
    #f2 = te.tps_fit_normals_exact_cvx(pts1, pts2, e1, e2, bend_coef=0.1, rot_coef=1e-5, normal_coef = 0.1, wt_n=None, nwsize=0.15, delta=0.002)
    
#    import IPython
#    IPython.embed()    
    mlab.figure(1, bgcolor=(0,0,0))
    mayavi_utils.plot_warping(f1, pts1, pts2, fine=False, draw_plinks=False)
    _,f1e2 = te.transformed_normal_direction(pts1, e1, f1, delta=0.0001)#np.asarray([tu.tps_jacobian(f2, pt, 2).dot(nm) for pt,nm in zip(pts1,e1)])
    test_normals_pts(f1.transform_points(pts1), f1e2, wsize=0.15,delta=0.15)
    test_normals_pts(pts2, e2, wsize=0.15,delta=0.15)
    #mlab.show()
    mlab.figure(2,bgcolor=(0,0,0))
    #mlab.clf()
    mayavi_utils.plot_warping(f2, pts1, pts2, fine=False, draw_plinks=False)
    _,f2e2 = te.transformed_normal_direction(pts1, e1, f2, delta=0.0001)#np.asarray([tu.tps_jacobian(f2, pt, 2).dot(nm) for pt,nm in zip(pts1,e1)])
    test_normals_pts(f2.transform_points(pts1), f2e2, wsize=0.15,delta=0.15)
    test_normals_pts(pts2, e2, wsize=0.15,delta=0.15)
    mlab.show()


mlab.figure(1, bgcolor=(0,0,0))
mayavi_utils.plot_warping(f1, pts1, pts2, fine=False, draw_plinks=False)

