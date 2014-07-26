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


pts1, pts2, e1, e2 = create_flap_points_normals(3.0,1,dim=3)

mlab.figure(1, bgcolor=(0,0,0))
mlab.show
