"""
Register point clouds to each other


arrays are named like name_abc
abc are subscripts and indicate the what that tensor index refers to

index name conventions:
    m: test point index
    n: training point index
    a: input coordinate
    g: output coordinate
    d: gripper coordinate
"""

from __future__ import division
import numpy as np
import scipy.spatial.distance as ssd
from rapprentice import tps, svds, math_utils
from tn_rapprentice import krig, tps as tn_tps
import tn_eval.tps_utils as turns
import numpy.linalg as nlg
from tn_rapprentice.tps import tps_eval, tps_grad, tps_fit3, tps_fit_regrot, tps_kernel_matrix, tps_cost
# from svds import svds
from tn_rapprentice import krig_utils as ku
import IPython as ipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tn_eval import tps_utils

class Transformation(object):
    """
    Object oriented interface for transformations R^d -> R^d
    """
    def transform_points(self, x_ma):
        raise NotImplementedError
    def compute_jacobian(self, x_ma):
        raise NotImplementedError        

        
    def transform_bases(self, x_ma, rot_mad, orthogonalize=True, orth_method = "cross"):
        """
        orthogonalize: none, svd, qr
        """

        grad_mga = self.compute_jacobian(x_ma)
        newrot_mgd = np.array([grad_ga.dot(rot_ad) for (grad_ga, rot_ad) in zip(grad_mga, rot_mad)])
        

        if orthogonalize:
            if orth_method == "qr": 
                newrot_mgd =  orthogonalize3_qr(newrot_mgd)
            elif orth_method == "svd":
                newrot_mgd = orthogonalize3_svd(newrot_mgd)
            elif orth_method == "cross":
                newrot_mgd = orthogonalize3_cross(newrot_mgd)
            else: raise Exception("unknown orthogonalization method %s"%orthogonalize)
        return newrot_mgd
        
    def transform_hmats(self, hmat_mAD):
        """
        Transform (D+1) x (D+1) homogenius matrices
        """
        hmat_mGD = np.empty_like(hmat_mAD)
        hmat_mGD[:,:3,3] = self.transform_points(hmat_mAD[:,:3,3])
        hmat_mGD[:,:3,:3] = self.transform_bases(hmat_mAD[:,:3,3], hmat_mAD[:,:3,:3])
        hmat_mGD[:,3,:] = np.array([0,0,0,1])
        return hmat_mGD
        
    def compute_numerical_jacobian(self, x_d, epsilon=0.0001):
        "numerical jacobian"
        x0 = np.asfarray(x_d)
        f0 = self.transform_points(x0)
        jac = np.zeros((len(x0), f0.shape[1], x0.shape[1]))
        dx = np.zeros(x0.shape[1])[None, :]
        for i in range(len(x0)):
            for j in range(len(dx)):
                dx[j] = epsilon
                jac[i,:,j] = (self.transform_points(x0[i][None,:] +dx) - self.transform_points(x0[i][None,:])) / epsilon
                dx[j] = 0.
        return jac.transpose()

class KrigingSpline(Transformation):
    def __init__(self, d=3, alpha = 1.5):
        self.alpha = alpha
        self.x_na = np.zeros((0,d))
        self.ex_na = np.zeros((0,d))
        self.exs = np.zeros((0,d))
        self.lin_ag = np.r_[np.zeros((1,d)), np.eye(d)]
        self.w_ng = np.zeros((0,d))
        self.trans_g = np.zeros(d)
    def transform_points(self, y_ng):
        y = krig.krig_eval(self.alpha, self.x_na, self.ex_na, self.exs, y_ng, self.w_ng, self.lin_ag, self.trans_g)
        return y
    def transform_normals(self, Eypts, Eys):
        y = krig.transform_normals1(self.alpha, self.x_na, self.ex_na, self.exs, Eypts, Eys,  self.w_ng, self.lin_ag)
        return y
    def compute_jacobian(self, x_ma):
        grad_mga = tn_tps.krig_grad(x_ma, self.ex_na, self.exs, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
        return grad_mga


class KrigingSplineLandmark(Transformation):
    def __init__(self, d=3, alpha = 1.5):
        self.alpha = alpha
        self.Xs = np.zeros((0,d))
        self.lin_ag = np.r_[np.zeros((1,d)), np.eye(d)]
        self.w_ng = np.zeros((0,d))
    def transform_points(self, Ys):
        Ys = krig.krig_eval_landmark(self.alpha, self.Xs, Ys, self.w_ng, self.lin_ag)
        return Ys
    def transform_normals(self, Eypts, Eys):
        y = krig.transform_normals1(self.alpha, self.x_na, self.ex_na, self.exs, Eypts, Eys, self.w_ng, self.lin_ag)

class ThinPlateSpline(Transformation):
    """
    members:
        x_na: centers of basis functions
        w_ng: 
        lin_ag: transpose of linear part, so you take x_na.dot(lin_ag)
        trans_g: translation part
    
    """
    def __init__(self, d=3):
        "initialize as identity"
        self.x_na = np.zeros((0,d))
        self.lin_ag = np.eye(d)
        self.trans_g = np.zeros(d)
        self.w_ng = np.zeros((0,d))

    def transform_points(self, x_ma):
        y_ng = tps.tps_eval(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
        return y_ng
    def compute_jacobian(self, x_ma):
        grad_mga = tps.tps_grad(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
        return grad_mga
        
class Affine(Transformation):
    def __init__(self, lin_ag, trans_g):
        self.lin_ag = lin_ag
        self.trans_g = trans_g
    def transform_points(self, x_ma):
        return x_ma.dot(self.lin_ag) + self.trans_g[None,:]  
    def compute_jacobian(self, x_ma):
        return np.repeat(self.lin_ag.T[None,:,:],len(x_ma), axis=0)
        
class Composition(Transformation):
    def __init__(self, fs):
        "applied from first to last (left to right)"
        self.fs = fs
    def transform_points(self, x_ma):
        for f in self.fs: x_ma = f.transform_points(x_ma)
        return x_ma
    def compute_jacobian(self, x_ma):
        grads = []
        for f in self.fs:
            grad_mga = f.compute_jacobian(x_ma)
            grads.append(grad_mga)
            x_ma = f.transform_points(x_ma)
        totalgrad = grads[0]
        for grad in grads[1:]:
            totalgrad = (grad[:,:,:,None] * totalgrad[:,None,:,:]).sum(axis=-2)
        return totalgrad

def fit_ThinPlateSpline(x_na, y_ng, bend_coef=.1, rot_coef = 1e-5, wt_n=None):
    """
    x_na: source cloud
    y_nd: target cloud
    smoothing: penalize non-affine part
    angular_spring: penalize rotation
    wt_n: weight the points        
    """
    d = x_na.shape[1]
    f = ThinPlateSpline(d)
    f.lin_ag, f.trans_g, f.w_ng = tps.tps_fit3(x_na, y_ng, bend_coef, rot_coef, wt_n)
    f.x_na = x_na
    return f        


def fit_KrigingSplineInterest(Xs, Epts, Exs, Ys, Eys, bend_coef = .01, normal_coef = 1, wt_n=None, alpha = 1.5, 
                    rot_coefs = 1e-5, interest_pts_inds = None):
    """
    Xs: landmark source cloud
    Epts: normal point source cloud
    Exs: normal values
    y_ng: landmark target cloud
    ey_ng: target normal point cloud
    Eys: target normal values
    wt_n: weight the points
    """
    d = Xs.shape[1]
    f = KrigingSpline(d, alpha)
    f.w_ng, f.trans_g, f.lin_ag = ku.krig_fit_interest(Xs, Ys, Epts, Exs, Eys, bend_coef = bend_coef, 
                                    normal_coef = normal_coef, wt_n = wt_n, rot_coefs = rot_coefs, interest_pts_inds = interest_pts_inds)
    f.x_na, f.ex_na, f.exs = Xs, Epts, Exs
    return f

def fit_KrigingSpline(Xs, Epts, Exs, Ys, Eys, bend_coef = .01, normal_coef = 1, wt_n=None, alpha = 1.5, 
                    rot_coefs = 1e-5, interest_pts_inds = None):
    """
    Xs: landmark source cloud
    Epts: normal point source cloud
    Exs: normal values
    y_ng: landmark target cloud
    ey_ng: target normal point cloud
    Eys: target normal values
    wt_n: weight the points
    """
    d = Xs.shape[1]
    f = KrigingSpline(d, alpha)
    f.w_ng, f.trans_g, f.lin_ag = ku.krig_fit1Normal(f.alpha, Xs, Ys, Epts, Exs, Eys, bend_coef = bend_coef, 
                                    normal_coef = normal_coef, wt_n = wt_n, rot_coefs = rot_coefs)
    f.x_na, f.ex_na, f.exs = Xs, Epts, Exs
    return f

def fit_KrigingSplineLandmark(Xs, Ys, bend_coef = 1e-6, alpha = 1.5, wt_n=None):
    """
    Xs: landmark source cloud
    Epts: normal point source cloud
    Exs: normal values
    y_ng: landmark target cloud
    ey_ng: target normal point cloud
    Eys: target normal values
    wt_n: weight the points
    """
    d = Xs.shape[1]
    f = KrigingSplineLandmark(d, alpha)
    f.w_ng, f.trans_g, f.lin_ag = ku.krig_fit3_landmark(f.alpha, Xs, Ys, bend_coef,  wt_n = None)
    f.x_na = Xs
    return f

def loglinspace(a,b,n):
    "n numbers between a to b (inclusive) with constant ratio between consecutive numbers"
    return np.exp(np.linspace(np.log(a),np.log(b),n))    
    
    
def tps_rpm(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg=1e-4,
            plotting = False, f_init = None, plot_cb = None, outlierprior = 1e-2):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    """
    _,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    if f_init is not None: 
        f = f_init  
    else:
        f = ThinPlateSpline(d)
        g = fit_KrigingSpline(x_nd, x_nd, x_nd, x_nd, x_nd)
        # f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)

    n,_ = x_nd.shape
    m,_ = y_md.shape

    x_priors = np.ones(n)*outlierprior    
    y_priors = np.ones(m)*outlierprior
    exs = np.ones((x_nd.shape))*2
    eys = np.ones((y_md.shape))   

    diff_tracker = np.zeros(1)


    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        corr_nm, r_n, _ = calc_correspondence_matrix(xwarped_nd, y_md, r=rads[i], p=.1, max_iter=10)
        wt_n = corr_nm.sum(axis=1)
        
        print i

        targ_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        
        if i >0:
            diff_tracker = np.r_[diff_tracker, np.sum(np.abs(xwarped_nd - targ_nd))]
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, targ_nd, corr_nm, wt_n, f)
        
        
        f = fit_ThinPlateSpline(x_nd, targ_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)
        #ipy.embed()
        
    
    #ipy.embed()
    return f, corr_nm


def tps_rpm_bij(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3, 
            plotting = False, plot_cb = None, x_weights = None, y_weights = None, outlierprior = .1, outlierfrac = 2e-1, vis_cost_xy = None):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    interest_pts are points in either scene where we want a lower prior of outliers
    x_weights: rescales the weights of the forward tps fitting of the last iteration
    y_weights: same as x_weights, but for the backward tps fitting
    """
    _,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)

    f = ThinPlateSpline(d)
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0) # align the medians
    # do a coarse search through rotations
    # fit_rotation(f, x_nd, y_md)
    
    g = ThinPlateSpline(d)
    g.trans_g = -f.trans_g

    # set up outlier priors for source and target scenes
    n, _ = x_nd.shape
    m, _ = y_md.shape

    x_priors = np.ones(n)*outlierprior    
    y_priors = np.ones(m)*outlierprior    

    # r_N = None
    
    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        ywarped_md = g.transform_points(y_md)
        
        fwddist_nm = ssd.cdist(xwarped_nd, y_md,'euclidean')
        invdist_nm = ssd.cdist(x_nd, ywarped_md,'euclidean')
        
        r = rads[i]
        prob_nm = np.exp( -(fwddist_nm + invdist_nm) / (2*r) )
        if vis_cost_xy != None:
            pi = np.exp( -vis_cost_xy )
            pi /= pi.max() # rescale the maximum probability to be 1. effectively, the outlier priors are multiplied by a visual prior of 1 (since the outlier points have a visual prior of 1 with any point)
            prob_nm *= pi
        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, .005) # edit final value to change outlier percentage
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)

        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)
        
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f)
        
        if i == (n_iter-1):
            if x_weights is not None:
                wt_n=wt_n*x_weights
            if y_weights is not None:
                wt_m=wt_m*y_weights
        f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)
        g = fit_ThinPlateSpline(y_md, ytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg)
        
        # add metadata of the transformation f
        f._corr = corr_nm
        f._bend_coef = regs[i]
        f._rot_coef = rot_reg
        f._wt_n = wt_n
    
    f._cost = tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, regs[i], wt_n=wt_n)/wt_n.mean()
    g._cost = tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ytarg_md, regs[i], wt_n=wt_m)/wt_m.mean()
    return f,corr_nm



def balance_matrix3(prob_nm, max_iter, row_priors, col_priors, outlierfrac, r_N = None):
    n,m = prob_nm.shape
    prob_NM = np.empty((n+1, m+1), 'f4')
    prob_NM[:n, :m] = prob_nm
    prob_NM[:n, m] = row_priors
    prob_NM[n, :m] = col_priors
    prob_NM[n, m] = np.sqrt(np.sum(row_priors)*np.sum(col_priors)) # this can `be weighted bigger weight = fewer outliers
    a_N = np.ones((n+1),'f4')
    a_N[n] = m*outlierfrac
    b_M = np.ones((m+1),'f4')
    b_M[m] = n*outlierfrac
    
    if r_N is None: r_N = np.ones(n+1,'f4')

    for _ in xrange(max_iter):
        c_M = b_M/r_N.dot(prob_NM)
        r_N = a_N/prob_NM.dot(c_M)

    prob_NM *= r_N[:,None]
    prob_NM *= c_M[None,:]
    
    return prob_NM[:n, :m], r_N, c_M


def balance_matrix(prob_nm, p, max_iter=20, ratio_err_tol=1e-3):
    n,m = prob_nm.shape
    pnoverm = (float(p)*float(n)/float(m))
    for _ in xrange(max_iter):
        colsums = pnoverm + prob_nm.sum(axis=0)        
        prob_nm /=  + colsums[None,:]
        rowsums = p + prob_nm.sum(axis=1)
        prob_nm /= rowsums[:,None]
        
        if ((rowsums-1).__abs__() < ratio_err_tol).all() and ((colsums-1).__abs__() < ratio_err_tol).all():
            break

    return prob_nm


def calc_correspondence_matrix(x_nd, y_md, r, p, max_iter=20):
    dist_nm = ssd.cdist(x_nd, y_md,'euclidean')
     
    prob_nm = np.exp(-dist_nm / r)
    # Seems to work better without **2
    # return balance_matrix(prob_nm, p=p, max_iter = max_iter, ratio_err_tol = ratio_err_tol)
    outlierfrac = 1e-1
    return balance_matrix33(prob_nm, max_iter, p, outlierfrac)


def close_to_perp(Epts, Exs, etarg, wt_n, perp_angle):
    _, d = Epts.shape
    theta = perp_angle*np.pi/180
    perp_thresh = np.cos(np.pi/2 - theta)
    Eptsn, Exsn, etargn = np.zeros((0,d)), np.zeros((0,d)), np.zeros((0,d))
    indices = np.zeros((Exs.shape[0], 1))
    for i in range(Exs.shape[0]):
        if not np.abs(np.dot(Exs[i], etarg[i])) < perp_thresh:
            indices[i,0] = 1
            Eptsn, Exsn, etargn = np.r_[Eptsn, np.array([Epts[i]])], np.r_[Exsn, np.array([Exs[i]])], np.r_[etargn, np.array([etarg[i]])]
            wt_n = np.r_[wt_n, wt_n[i]]

    return Eptsn, Exsn, etargn, wt_n, indices

def rotation(theta):
    tx,ty,tz = theta
 
    Rx = np.array([[1,0,0], [0, cos(tx), -sin(tx)], [0, sin(tx), cos(tx)]])
    Ry = np.array([[cos(ty), 0, -sin(ty)], [0, 1, 0], [sin(ty), 0, cos(ty)]])
    Rz = np.array([[cos(tz), -sin(tz), 0], [sin(tz), cos(tz), 0], [0,0,1]])
 
    return np.dot(Rx, np.dot(Ry, Rz))

def rotate_point_cloud3d(pcloud, angle):
    ty = angle*np.pi/180
    Ry = np.array([[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]])
    Rpc = np.zeros((pcloud.shape))
    for i in range(len(pcloud)):
        Rpc[i] = Ry.dot(pcloud[i])
    return Rpc

def rotate_point_cloud2d(pcloud, angle):
    ty = angle*np.pi/180
    Ry = np.array([[np.cos(ty), -np.sin(ty)], [np.sin(ty), np.cos(ty)]])
    Rpc = np.zeros((pcloud.shape))
    for i in range(len(pcloud)):
        Rpc[i] = Ry.dot(pcloud[i])
    return Rpc

def rotate_point2d(pt, angle):
    ty = angle*np.pi/180
    Ry = np.array([[np.cos(ty), -np.sin(ty)], [np.sin(ty), np.cos(ty)]])
    Rp = Ry.dot(pt)
    return Rp

def krig_obj(f, x_na, y_ng, exs, eys, bend_coef, rot_coef, wt_n):
     # expand these
    n,d = x_na.shape
    bend_coefs = np.ones(d) * bend_coef if np.isscalar(bend_coef) else bend_coef
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    if wt_n is None: wt_n = np.ones(len(x_na))
    if wt_n.ndim == 1:
        wt_n = wt_n[:,None]
    if wt_n.shape[1] == 1:
        wt_n = np.tile(wt_n, (1,d))
    
    K_nn = ku.krig_kernel_mat(1.5, x_na, f.ex_na, exs)
    cost = 0
    # matching cost
    cost += np.linalg.norm((f.transform_points(x_na) - y_ng) * np.sqrt(wt_n[:n]))**2
    # same as (np.square(np.apply_along_axis(np.linalg.norm, 1, f.transform_points(x_na) - y_ng)) * wt_n).sum()
    # bending cost
    cost += np.trace(np.diag(bend_coefs).dot(f.w_ng.T.dot(K_nn.dot(f.w_ng))))
    cost += np.linalg.norm((f.transform_normals(x_na, exs) - eys) * np.sqrt(wt_n[n:]))**2

    # rotation cost
    #cost += np.trace((f.lin_ag - np.eye(d)).T.dot(np.diag(rot_coefs).dot((f.lin_ag - np.eye(d)))))
#     # constants
#     cost -= np.linalg.norm(y_ng * np.sqrt(wt_n))**2
#     cost -= np.trace(np.diag(rot_coefs))
    return cost

def tps_obj(f, x_na, y_ng, bend_coef, rot_coef, wt_n):
    # expand these
    _,d = x_na.shape
    bend_coefs = np.ones(d) * bend_coef if np.isscalar(bend_coef) else bend_coef
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    if wt_n is None: wt_n = np.ones(len(x_na))
    if wt_n.ndim == 1:
        wt_n = wt_n[:,None]
    if wt_n.shape[1] == 1:
        wt_n = np.tile(wt_n, (1,d))
    
    K_nn = tps.tps_kernel_matrix(x_na)
    _,d = x_na.shape
    cost = 0
    # matching cost
    cost += np.linalg.norm((f.transform_points(x_na) - y_ng) * np.sqrt(wt_n))**2
    # same as (np.square(np.apply_along_axis(np.linalg.norm, 1, f.transform_points(x_na) - y_ng)) * wt_n).sum()
    # bending cost
    cost += np.trace(np.diag(bend_coefs).dot(f.w_ng.T.dot(K_nn.dot(f.w_ng))))
    # rotation cost
    cost += np.trace((f.lin_ag - np.eye(d)).T.dot(np.diag(rot_coefs).dot((f.lin_ag - np.eye(d)))))

    return cost


def EM_step(f, x_nd, y_md, outlierfrac, Temp, bend_coef, rot_reg, outlierprior, curve_cost = None, beta = 1):
    n,_ = x_nd.shape
    m,_ = y_md.shape
    x_priors = np.ones(n)*outlierprior    
    y_priors = np.ones(m)*outlierprior

    xwarped_nd = f.transform_points(x_nd)
    dist_nm = ssd.cdist(xwarped_nd, y_md,'sqeuclidean')
    T = Temp
    prob_nm = np.exp( -dist_nm / T )

    beta = beta
    if curve_cost != None:
        pi = np.exp(-beta*curve_cost)
        pi /= pi.max() # we can do better I think
        prob_nm *= pi
        #ipy.embed()

    corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, outlierfrac) # edit final value to change outlier percentage
    corr_nm += 1e-9    
    wt_n = corr_nm.sum(axis=1)

    targ_nd = (corr_nm/wt_n[:,None]).dot(y_md)
      
    f = fit_ThinPlateSpline_corr(x_nd, y_md, corr_nm, bend_coef, rot_reg)
    return f, corr_nm

def fit_ThinPlateSpline_corr(x_nd, y_md, corr_nm, bend_coef, rot_reg, x_weights = None):
    wt_n = corr_nm.sum(axis=1)
    if np.any(wt_n == 0):
        inlier = wt_n != 0
        x_nd = x_nd[inlier,:]
        wt_n = wt_n[inlier,:]
        x_weights = x_weights[inlier]
        xtarg_nd = (corr_nm[inlier,:]/wt_n[:,None]).dot(y_md)
    else:
        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
    if x_weights is not None:
        if x_weights.ndim > 1:
            wt_n=wt_n[:,None]*x_weights
        else:
            wt_n=wt_n*x_weights
    f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = bend_coef, wt_n = wt_n, rot_coef = rot_reg)
    f._bend_coef = bend_coef
    f._wt_n = wt_n
    f._rot_coef = rot_reg
    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, bend_coef, wt_n=wt_n)/wt_n.mean()
    return f

def compute_curvature_cost(x_nd, y_md, orig_source, orig_target, wsize):
    x_curves = tps_utils.find_all_curvatures(x_nd, orig_source, wsize)
    y_curves = tps_utils.find_all_curvatures(y_md, orig_target, wsize)

    x_curves = np.c_[x_curves, np.zeros((len(x_curves), 1))]
    y_curves = np.c_[y_curves, np.zeros((len(y_curves), 1))]

    curve_cost = ssd.cdist(x_curves, y_curves, 'cityblock')

    return curve_cost
def compute_normals_cost(x_nd, y_md, orig_source, orig_target, wsize):
    x_normals = tps_utils.find_all_normals_naive(x_nd, orig_source, wsize)
    y_normals = tps_utils.find_all_normals_naive(y_md, orig_target, wsize)

    normals_cost = ssd.cdist(x_normals, y_normals, 'sqeuclidean')

    return normals_cost

def tps_rpm_curvature_rpm_joint(x_nd, y_md, orig_source = None, orig_target = None, n_iter=20, T_init=.1,  T_final=.01, bend_init=.1, bend_final=.01,
                     rot_reg = 1e-5,  outlierfrac = 1e-2, wsize = .1, EM_iter = 5, f_init = None, outlierprior = .1, beta = 1, plotting = False, angle = 0,
                     square_size = 0, circle_rad = 0):
    _,d=x_nd.shape
    Ts = loglinspace(T_init, T_final, n_iter)
    Bs = loglinspace(bend_init, bend_final, n_iter)
    fc = ThinPlateSpline(d)
    f = ThinPlateSpline(d)

    if orig_source is None:
        orig_source = x_nd
    if orig_target is None:
        orig_target = y_md

    x0s, x1s, x2s, x3s, x4s = np.array([0,0]), np.array([1,0]), np.array([1,1]), np.array([.5, 1.5,]), np.array([0,1])
    x0sr, x1sr, x2sr, x3sr, x4sr = rotate_point2d(x0s, angle),rotate_point2d(x1s, angle),rotate_point2d(x2s, angle),rotate_point2d(x3s, angle),rotate_point2d(x4s, angle)

    curve_cost = compute_curvature_cost(x_nd, y_md, orig_source, orig_target, wsize)

    #plt.ion()
    #fig1 = plt.figure()
    #ax1 = fig1.add_subplot(111, projection='3d')

    number_points=30

    for i in xrange(n_iter):
        for j in range(EM_iter):
            print i, j
            fc, corr_nm = EM_step(fc, x_nd, y_md, outlierfrac, Ts[i], Bs[i], rot_reg, outlierprior, curve_cost = curve_cost, beta = beta)
            f, gcorr_nm = EM_step(f, x_nd, y_md, outlierfrac, Ts[i], Bs[i], rot_reg, outlierprior)

        #ipy.embed()
        if plotting and i%plotting==0:            #plt.clf()
            plot_house(fc, x0sr, x1sr, x2sr, x3sr, x4sr, 30, corr_nm, y_md)
            plot_house(f, x0sr, x1sr, x2sr, x3sr, x4sr, 30, gcorr_nm, y_md)
            
            #plot_corr(corr_nm, y_md)   
            plt.show()   
            
    #ipy.embed()
    return fc, f
#@profile
def tps_rpm_EM(x_nd, y_md,  n_iter=20, T_init=.1,  T_final=.01, bend_init=.1, bend_final=.01, rot_reg = 1e-5, 
             outlierfrac = 1e-2, EM_iter = 5, f_init = None, outlierprior = .1, plotting = False, angle = 0,
             square_size = 0, circle_rad = 0):
    _,d=x_nd.shape
    Ts = loglinspace(T_init, T_final, n_iter)
    Bs = loglinspace(bend_init, bend_final, n_iter)
    if f_init is not None: 
        f = f_init  
    else:
        f = ThinPlateSpline(d)
        # f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)
    x0s, x1s, x2s, x3s, x4s = np.array([0,0]), np.array([1,0]), np.array([1,1]), np.array([.5, 1.5,]), np.array([0,1])
    x0sr, x1sr, x2sr, x3sr, x4sr = rotate_point2d(x0s, angle),rotate_point2d(x1s, angle),rotate_point2d(x2s, angle),rotate_point2d(x3s, angle),rotate_point2d(x4s, angle)

    for i in xrange(n_iter):
        for j in range(EM_iter):
            print i,j
            f, corr_nm = EM_step(f, x_nd, y_md, outlierfrac, Ts[i], Bs[i], rot_reg, outlierprior)
        if plotting and i%plotting==0:
            plot_grabs_two(f, x_nd, angle)
            #if square_size and circle_rad:
            #    plot_box_circle(f, square_size, circle_rad, angle, corr = corr_nm, Y = y_md)
            #elif rope_plotting:
            #    plot_house(f, x0sr, x1sr, x2sr, x3sr, x4sr, 30, corr_nm, y_md)
            plt.show()    
    #ipy.embed()
    return f, corr_nm

#@profile
def tps_rpm_curvature_prior1(x_nd, y_md, orig_source = None, orig_target = None, n_iter=20, T_init=.1,  T_final=.01, bend_init=.1, bend_final=.01,
                     rot_reg = 1e-5,  outlierfrac = 1e-2, wsize = .1, EM_iter = 5, f_init = None, outlierprior = .1, beta = 1, plotting = False, angle = 0,
                     square_size = 0, circle_rad = 0):
    _,d=x_nd.shape
    Ts = loglinspace(T_init, T_final, n_iter)
    Bs = loglinspace(bend_init, bend_final, n_iter)
    if f_init is not None: 
        f = f_init  
    else:
        f = ThinPlateSpline(d)
        # f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)

    if orig_source is None:
        orig_source = x_nd
    if orig_target is None:
        orig_target = y_md

    x0s, x1s, x2s, x3s, x4s = np.array([0,0]), np.array([1,0]), np.array([1,1]), np.array([.5, 1.5,]), np.array([0,1])
    x0sr, x1sr, x2sr, x3sr, x4sr = rotate_point2d(x0s, angle),rotate_point2d(x1s, angle),rotate_point2d(x2s, angle),rotate_point2d(x3s, angle),rotate_point2d(x4s, angle)

    curve_cost = compute_curvature_cost(x_nd, y_md, orig_source, orig_target, wsize)

    #plt.ion()
    #fig1 = plt.figure()
    #ax1 = fig1.add_subplot(111, projection='3d')

    number_points=30

    for i in xrange(n_iter):
        for j in range(EM_iter):
            print i, j
            f, corr_nm = EM_step(f, x_nd, y_md, outlierfrac, Ts[i], Bs[i], rot_reg, outlierprior, curve_cost = curve_cost, beta = beta)
        #ipy.embed()
        if plotting and i%plotting==0:
            plot_grabs_two(f, x_nd, angle)

            #if square_size and circle_rad:
            #    plot_box_circle(f, square_size, circle_rad, angle, corr = corr_nm, Y = y_md)
            #else:
            #    plot_house(f, x0sr, x1sr, x2sr, x3sr, x4sr, 30, corr_nm, y_md)
            plt.show()
  
            
    #ipy.embed()
    return f, corr_nm

def tps_rpm_curvature_prior2(x_nd, y_md, orig_source = None, orig_target = None, n_iter=20, T_init=.1,  T_final=.01, bend_init=.1, bend_final=.01,
                     rot_reg = 1e-5,  outlierfrac = 1e-2, wsize = .1, EM_iter = 5, f_init = None, outlierprior = .1, beta = 1, plotting = False, angle = 0,
                     square_size = 0, circle_rad = 0):
    _,d=x_nd.shape
    Ts = loglinspace(T_init, T_final, n_iter)
    Bs = loglinspace(bend_init, bend_final, n_iter)
    if f_init is not None: 
        f = f_init  
    else:
        f = ThinPlateSpline(d)
        # f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)

    x0s, x1s, x2s, x3s, x4s = np.array([0,0]), np.array([1,0]), np.array([1,1]), np.array([.5, 1.5,]), np.array([0,1])
    x0sr, x1sr, x2sr, x3sr, x4sr = rotate_point2d(x0s, angle),rotate_point2d(x1s, angle),rotate_point2d(x2s, angle),rotate_point2d(x3s, angle),rotate_point2d(x4s, angle)

    if orig_source is None:
        orig_source = x_nd
    if orig_target is None:
        orig_target = y_md

    for i in xrange(n_iter):
        for j in range(EM_iter):
            print i, j
            f, corr_nm = EM_step_curves(f, x_nd, y_md, outlierfrac, Ts[i], Bs[i], rot_reg, outlierprior, beta = beta)
        if plotting and i%plotting==0:
            if square_size and circle_rad:
                plot_box_circle(f, square_size, circle_rad, angle, corr = corr_nm, Y = y_md)
            else:
                plot_house(f, x0sr, x1sr, x2sr, x3sr, x4sr, 30, corr_nm, y_md)
            plt.show()
    #ipy.embed()
    return f, corr_nm


def tps_rpm_normals_prior(x_nd, y_md, orig_source = None, orig_target = None, n_iter=20, T_init=.1,  T_final=.01, bend_init=.1, bend_final=.01,
                     rot_reg = 1e-5,  outlierfrac = 1e-2, wsize = .1, EM_iter = 5, f_init = None, outlierprior = .1, beta = 1, plotting = False, angle = 0,
                     normal_init = 5, normal_final = .5, square_size = 0, circle_rad = 0):
    
    _,d=x_nd.shape
    Ts = loglinspace(T_init, T_final, n_iter)
    Bs = loglinspace(bend_init, bend_final, n_iter)
    Ns = loglinspace(normal_init, normal_final, n_iter)
    if f_init is not None: 
        f = f_init  
    else:
        f = ThinPlateSpline(d)
        # f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)

    x0s, x1s, x2s, x3s, x4s = np.array([0,0]), np.array([1,0]), np.array([1,1]), np.array([.5, 1.5,]), np.array([0,1])
    x0sr, x1sr, x2sr, x3sr, x4sr = rotate_point2d(x0s, angle),rotate_point2d(x1s, angle),rotate_point2d(x2s, angle),rotate_point2d(x3s, angle),rotate_point2d(x4s, angle)

    if orig_source is None:
        orig_source = x_nd
    if orig_target is None:
        orig_target = y_md

    
    for i in xrange(n_iter):
        for j in range(EM_iter):
            print i, j
            f, corr_nm = EM_step_normals(f, x_nd, y_md, outlierfrac, Ts[i], Bs[i], Ns[i], rot_reg, outlierprior, beta = beta)
        if plotting and i%plotting==0:
            if square_size and circle_rad:
                plot_box_circle(f, square_size, circle_rad, angle, corr = corr_nm, Y = y_md)
            else:
                plot_house(f, x0sr, x1sr, x2sr, x3sr, x4sr, 30, corr_nm, y_md)
            plt.show()
    #ipy.embed()
    return f, corr_nm

def tps_rpm_normals_curves(x_nd, y_md, orig_source = None, orig_target = None, n_iter=20, T_init=.1,  T_final=.01, bend_init=.1, bend_final=.01,
                     rot_reg = 1e-5,  outlierfrac = 1e-2, wsize = .1, EM_iter = 5, f_init = None, outlierprior = .1, beta = 1, plotting = False, angle = 0,
                     normal_init = 5, normal_final = .5, square_size = 0, circle_rad = 0):
    
    _,d=x_nd.shape
    Ts = loglinspace(T_init, T_final, n_iter)
    Bs = loglinspace(bend_init, bend_final, n_iter)
    Ns = loglinspace(normal_init, normal_final, n_iter)
    if f_init is not None: 
        f = f_init  
    else:
        f = ThinPlateSpline(d)
        # f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)

    x0s, x1s, x2s, x3s, x4s = np.array([0,0]), np.array([1,0]), np.array([1,1]), np.array([.5, 1.5,]), np.array([0,1])
    x0sr, x1sr, x2sr, x3sr, x4sr = rotate_point2d(x0s, angle),rotate_point2d(x1s, angle),rotate_point2d(x2s, angle),rotate_point2d(x3s, angle),rotate_point2d(x4s, angle)

    if orig_source is None:
        orig_source = x_nd
    if orig_target is None:
        orig_target = y_md

    for i in xrange(n_iter):
        for j in range(EM_iter):
            print i, j
            f, corr_nm = EM_step_normals_curves(f, x_nd, y_md, outlierfrac, Ts[i], Bs[i], Ns[i], rot_reg, outlierprior, beta = beta)
        if plotting and i%plotting==0:
            if square_size and circle_rad:
                plot_box_circle(f, square_size, circle_rad, angle, corr = corr_nm, Y = y_md)
            else:
                plot_house(f, x0sr, x1sr, x2sr, x3sr, x4sr, 30, corr_nm, y_md)
            plt.show()
    #ipy.embed()
    return f, corr_nm

#@profile
def EM_step(f, x_nd, y_md, outlierfrac, temp, bend_coef, rot_reg, outlierprior, curve_cost = None, beta = 1):
    n,_ = x_nd.shape
    m,_ = y_md.shape
    x_priors = np.ones(n)*outlierprior    
    y_priors = np.ones(m)*outlierprior

    xwarped_nd = f.transform_points(x_nd)
    dist_nm = ssd.cdist(xwarped_nd, y_md,'sqeuclidean')
    T = temp
    prob_nm = np.exp( -dist_nm / T )

    beta = beta
    if curve_cost != None:
        pi = np.exp(-beta*curve_cost)
        pi /= pi.max() # we can do better I think
        prob_nm *= pi
        #ipy.embed()

    corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, outlierfrac) # edit final value to change outlier percentage
    corr_nm += 1e-9    
    wt_n = corr_nm.sum(axis=1)

    targ_nd = (corr_nm/wt_n[:,None]).dot(y_md)
      
    f = fit_ThinPlateSpline_corr(x_nd, y_md, corr_nm, bend_coef, rot_reg)
    return f, corr_nm

def EM_step_curves(f, x_nd, y_md, outlierfrac, temp, bend_coef, rot_reg, outlierprior, curve_cost = None, beta = 1):
    n,_ = x_nd.shape
    m,_ = y_md.shape
    x_priors = np.ones(n)*outlierprior    
    y_priors = np.ones(m)*outlierprior

    xwarped_nd = f.transform_points(x_nd)
    dist_nm = ssd.cdist(xwarped_nd, y_md,'sqeuclidean')
    T = temp
    prob_nm = np.exp( -dist_nm / T )

    
    curve_cost = compute_curvature_cost(xwarped_nd, y_md, y_md, y_md, .1)
    #ipy.embed()

    beta = beta
    pi = np.exp(-beta*curve_cost)
    pi /= pi.max() # we can do better I think
    try: 
        prob_nm *= pi
    except:
        print None
        #ipy.embed()
    corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, outlierfrac) # edit final value to change outlier percentage
    corr_nm += 1e-9    
    wt_n = corr_nm.sum(axis=1)

    targ_nd = (corr_nm/wt_n[:,None]).dot(y_md)
      
    f = fit_ThinPlateSpline_corr(x_nd, y_md, corr_nm, bend_coef, rot_reg)
    return f, corr_nm



def EM_step_normals(f, x_nd, y_md, outlierfrac, temp, bend_coef, normal_weight, rot_reg, outlierprior, curve_cost = None, beta = 1):
    n,_ = x_nd.shape
    m,_ = y_md.shape
    x_priors = np.ones(n)*outlierprior    
    y_priors = np.ones(m)*outlierprior

    xwarped_nd = f.transform_points(x_nd)
    dist_nm = ssd.cdist(xwarped_nd, y_md,'sqeuclidean')
    T = temp
    prob_nm = np.exp( -dist_nm / T )

    normals_cost = compute_normals_cost(xwarped_nd, y_md, y_md, y_md, .1)
    
    beta = beta
    pi = np.exp(-beta*normals_cost/normal_weight)
    #pi /= pi.max() # we can do better I think
    prob_nm *= pi
    corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, outlierfrac) # edit final value to change outlier percentage
    corr_nm += 1e-9    
    wt_n = corr_nm.sum(axis=1)

    targ_nd = (corr_nm/wt_n[:,None]).dot(y_md)
      
    f = fit_ThinPlateSpline_corr(x_nd, y_md, corr_nm, bend_coef, rot_reg)
    return f, corr_nm

def EM_step_normals_curves(f, x_nd, y_md, outlierfrac, temp, bend_coef, normal_weight, rot_reg, outlierprior, curve_cost = None, beta = 1):
    n,_ = x_nd.shape
    m,_ = y_md.shape
    x_priors = np.ones(n)*outlierprior    
    y_priors = np.ones(m)*outlierprior

    xwarped_nd = f.transform_points(x_nd)
    dist_nm = ssd.cdist(xwarped_nd, y_md,'sqeuclidean')
    T = temp
    prob_nm = np.exp( -dist_nm / T )
    

    normals_cost = compute_normals_cost(xwarped_nd, y_md, y_md, y_md, .1)
    
    
    beta = beta
    pi = np.exp(-beta*normals_cost/normal_weight)
    #pi /= pi.max() # we can do better I think
    prob_nm *= pi
    
    curve_cost = compute_curvature_cost(xwarped_nd, y_md, y_md, y_md, .1)
    #ipy.embed()

    beta = beta
    pi2 = np.exp(-beta*curve_cost)
    #pi2 /= pi2.max() # we can do better I think

    corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, outlierfrac) # edit final value to change outlier percentage
    corr_nm += 1e-9    
    wt_n = corr_nm.sum(axis=1)

    targ_nd = (corr_nm/wt_n[:,None]).dot(y_md)
      
    f = fit_ThinPlateSpline_corr(x_nd, y_md, corr_nm, bend_coef, rot_reg)

    #print np.sum(np.min(normals_cost, axis = 1))/len(normals_cost)
    #print np.sum(np.min(curve_cost, axis=1))/len(curve_cost)
    #print np.sum(np.min(dist_nm, axis=1))/len(dist_nm)
    #print np.sum(np.max(prob_nm, axis=1))/len(prob_nm)

    ewarped = tps_utils.find_all_normals_naive(f.transform_points(x_nd), y_md, wsize =.1)
    es = tps_utils.find_all_normals_naive(y_md, y_md, wsize = .1)
    print np.sum(np.abs(ewarped - es))/len(es)
    print np.sum(np.abs(f.transform_points(x_nd) - y_md))/len(y_md)

    return f, corr_nm

def plot_house(f1, x0s, x1s, x2s, x3s, x4s, number_points, corr = None, Y=None):
    bottom_row = np.c_[np.linspace(x0s[0], x1s[0], number_points), np.linspace(x0s[1], x1s[1], number_points)]
    right_column = np.c_[np.linspace(x1s[0], x2s[0], number_points), np.linspace(x1s[1], x2s[1], number_points)]
    right_diagonal = np.c_[np.linspace(x2s[0], x3s[0], number_points), np.linspace(x2s[1], x3s[1], number_points)]
    left_diagonal = np.c_[np.linspace(x3s[0], x4s[0], number_points), np.linspace(x3s[1], x4s[1], number_points)]
    left_column = np.c_[np.linspace(x4s[0], x0s[0], number_points), np.linspace(x4s[1], x0s[1], number_points)]


    f1bottom_row = f1.transform_points(bottom_row)
    f1right_column = f1.transform_points(right_column)
    f1right_diagonal = f1.transform_points(right_diagonal)
    f1left_diagonal = f1.transform_points(left_diagonal)
    f1left_column = f1.transform_points(left_column)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(f1bottom_row[:,0], f1bottom_row[:,1], color = 'g')
    ax1.scatter(f1right_column[:,0], f1right_column[:,1], color = 'y')
    ax1.scatter(f1left_column[:,0], f1left_column[:,1], color = 'r')
    ax1.scatter(f1left_diagonal[:,0], f1left_diagonal[:,1], color = 'b')
    ax1.scatter(f1right_diagonal[:,0], f1right_diagonal[:,1], color = 'g')

    if corr is not None and Y is not None:
        MY = corr.dot(Y)
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection = '3d')
        ax1.scatter(MY[:,0], MY[:,1], np.ones((len(MY), 1)), color = 'b')
    #plt.clf()
    #fig1.canvas.draw()

def plot_box_circle(f, square_size, circle_rad, angle, corr=None, Y=None, square_xtrans = 0, square_ytrans = 0, circle_xtrans = 0, circle_ytrans = 1.5, number_points = 50):
    from tn_testing.test_tps import gen_circle_points
    square_size = float(square_size)

    x0 = np.array([-square_size/2, -square_size/2])
    x1 = np.array([square_size/2, -square_size/2])
    x2 = np.array([square_size/2, square_size/2])
    x3 = np.array([-square_size/2, square_size/2])

    bottom = np.c_[np.linspace(x0[0], x1[0], number_points), np.linspace(x0[1], x1[1], number_points)]
    right = np.c_[np.linspace(x1[0], x2[0], number_points), np.linspace(x1[1], x2[1], number_points)]
    left = np.c_[np.linspace(x0[0], x3[0], number_points), np.linspace(x0[1], x3[1], number_points)]
    top = np.c_[np.linspace(x3[0], x2[0], number_points), np.linspace(x3[1], x2[1], number_points)]

    square = np.r_[bottom, right, left, top]
    circle = gen_circle_points(circle_rad, number_points)
    squarer = rotate_point_cloud2d(square, angle)
    circler = rotate_point_cloud2d(circle, angle)
    line_r = rotate_point2d(np.array([circle_xtrans, circle_ytrans]), angle)

    bottomr = rotate_point_cloud2d(bottom, angle)
    rightr = rotate_point_cloud2d(right, angle)
    leftr = rotate_point_cloud2d(left, angle)
    topr = rotate_point_cloud2d(top, angle)


    square_trans = squarer + np.array([square_xtrans, square_ytrans])
    bottom_trans = bottomr + np.array([square_xtrans, square_ytrans])
    right_trans = rightr + np.array([square_xtrans, square_ytrans])
    top_trans = topr + np.array([square_xtrans, square_ytrans])
    left_trans = leftr + np.array([square_xtrans, square_ytrans])
    circle_trans = circler + line_r

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fleft = f.transform_points(left_trans)
    ftop = f.transform_points(top_trans)
    fright = f.transform_points(right_trans)
    fbottom = f.transform_points(bottom_trans)

    fcircle = f.transform_points(circle_trans)
    ax.scatter(fleft[:,0], fleft[:,1], color = 'r')
    ax.scatter(ftop[:,0], ftop[:,1], color = 'y')
    ax.scatter(fright[:,0], fright[:,1], color = 'g')
    ax.scatter(fbottom[:,0], fbottom[:,1], color = 'b')

    ax.scatter(fcircle[:,0], fcircle[:,1], color = 'b')

    if corr is not None and Y is not None:
        MY = corr.dot(Y)
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(MY[:,0], MY[:,1], np.ones((len(MY), 1)), color = 'b')

def plot_grabs_two(f, pts1, angle):
    pts1r = rotate_point_cloud2d(pts1, angle)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    fpts1r = f.transform_points(pts1r)

    ax.scatter(fpts1r[:,0], fpts1r[:,1])

def plot_corr(corr, Y):
    MY = corr.dot(Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(MY[:,0], MY[:,1], color = 'b')
    

def main():
    from tn_testing.test_tps import gen_half_sphere, gen_half_sphere_pulled_in, gen_house, gen_box_circle
    from tn_eval.tps_utils import find_all_normals_naive
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from tn_testing import grabs_two


    """
    pts1 = gen_half_sphere(1, 30)
    pts1r = np.random.permutation(pts1)
    pts2 = gen_half_sphere_pulled_in(1, 30, 20, .5)
    e1 = find_all_normals_naive(pts1, wsize = .7,flip_away=True)
    e1r = find_all_normals_naive(pts1r, wsize = .7, flip_away=True)
    e2 = find_all_normals_naive(pts2, wsize = .7, flip_away=True)
    
    pts1rr = rotate_point_cloud3d(pts1, 90)
    e1rr = find_all_normals_naive(pts1rr, wsize=.7, flip_away=True)

    pts2r = rotate_point_cloud3d(pts1, 90)
    """
    number_points = 30
    square_size1 = 1
    square_size2 = 1
    circle_rad1 = .5
    circle_rad2 = .5

    angle = 0

    x0s, x1s, x2s, x3s, x4s = np.array([0,0]), np.array([1,0]), np.array([1,1]), np.array([.5, 3]), np.array([0,1])
    #pts1 = gen_house(x0s, x1s, x2s, x3s, x4s, number_points)
    #pts1 = gen_box_circle(square_size1, circle_rad1, number_points = number_points)
    #pts2 = gen_box_circle(square_size2, circle_rad2, number_points = number_points)
    pts1 = grabs_two.old_cloud[:,:3]
    pts2 = grabs_two.new_cloud[:,:3]
    big_old_cloud = np.random.random((554, 3))
    big_new_cloud = np.random.random((8972, 3))
    #pts1r = np.random.permutation(pts1)
    pts1r = rotate_point_cloud3d(pts1, angle)
    #x0sr, x1sr, x2sr, x3sr, x4sr = rotate_point2d(x0s, angle),rotate_point2d(x1s, angle),rotate_point2d(x2s, angle),rotate_point2d(x3s, angle),rotate_point2d(x4s, angle)

    #x0t, x1t, x2t, x3t, x4t = np.array([0,0]), np.array([1,0]), np.array([1,.5]), np.array([.5, 1.5]), np.array([0,.5])
    #pts2 = gen_house(x0t, x1t, x2t, x3t, x4t, number_points)

    EM_iter = 1

    beta = 2e1 #20 works for 90 rotation
    wsize = .1
    plotting = 0
    
    T_init = .04
    T_final = .00004
    bend_init = 1 # 1e2 works for 90 rotation
    bend_final = .001
    normal_init = 1
    normal_final = 1

    #ipy.embed()

    n = 1, 4

    if 1 in n:
        f1 , corr1 = tps_rpm_curvature_prior1(pts1r, pts2, orig_source = big_old_cloud, orig_target = big_new_cloud, n_iter = 20, EM_iter = EM_iter, T_init = T_init, T_final = T_final, bend_init = bend_init, bend_final = bend_final, wsize = wsize, beta = beta, plotting = plotting, angle = angle, square_size = square_size1, circle_rad = circle_rad1)
        #plot_house(f1, x0sr, x1sr, x2sr, x3sr, x4sr, number_points)
        #plot_box_circle(f1, square_size1, circle_rad1, angle, number_points = number_points)
        #plot_grabs_two(f1, pts1r, angle)

    if 2 in n:
        f2 , corr2 = tps_rpm_curvature_prior2(pts1r, pts2, n_iter = 20, EM_iter = EM_iter, T_init = T_init, T_final = T_final, bend_init = bend_init, bend_final = bend_final, wsize = wsize, beta = beta, plotting = plotting, angle = angle, square_size = square_size1, circle_rad = circle_rad1)
        #plot_house(f2, x0s, x1sr, x2sr, x3sr, x4sr, number_points)
        #plot_box_circle(f2, square_size1, circle_rad1, angle, number_points = number_points)
        plot_grabs_two(f2, pts1r, angle)

    if 3 in n:
        f3, corr3 = tps_rpm_bij(pts1r,pts2, reg_init = T_init, rad_init = bend_init)
        #plot_house(f3, x0sr, x1sr, x2sr, x3sr, x4sr, number_points)
        #plot_box_circle(f3, square_size1, circle_rad1, angle, number_points = number_points)
        plot_grabs_two(f3, pts1r, angle)

    if 4 in n:
        f4,corr4 = tps_rpm_EM(pts1r, pts2, n_iter = 20, EM_iter = EM_iter, T_init = T_init, T_final = T_final, bend_init = bend_init, bend_final = bend_final, plotting = plotting, angle = angle, square_size = square_size1, circle_rad = circle_rad1)
        #plot_house(f4, x0sr, x1sr, x2sr, x3sr, x4sr, number_points)
        #plot_box_circle(f4, square_size1, circle_rad1, angle, number_points = number_points)
        #plot_grabs_two(f4, pts1r, angle)

    if 5 in n:
        f5, corr5 = tps_rpm_normals_prior(pts1r, pts2, n_iter = 20, EM_iter = EM_iter, T_init = T_init, T_final = T_final, bend_init = bend_init, bend_final = bend_final, plotting = plotting, angle = angle, normal_init = normal_init, normal_final = normal_final,square_size = square_size1, circle_rad = circle_rad1)
        #plot_house(f5, x0sr, x1sr, x2sr, x3sr, x4sr, number_points)
        #plot_box_circle(f5, square_size1, circle_rad1, angle, number_points = number_points)
        plot_grabs_two(f5, pts1r, angle)

    if 6 in n:
        f6, corr6 = tps_rpm_normals_curves(pts1r, pts2, n_iter = 20, EM_iter = EM_iter, T_init = T_init, T_final = T_final, bend_init = bend_init, bend_final = bend_final, plotting = plotting, angle = angle, normal_init = normal_init, normal_final = normal_final, beta = beta, square_size = square_size1, circle_rad = circle_rad1)
        #plot_house(f6, x0sr, x1sr, x2sr, x3sr, x4sr, number_points)
        #plot_box_circle(f6, square_size1, circle_rad1, angle, number_points = number_points)
        plot_grabs_two(f6, pts1r, angle)

    #hc , h = tps_rpm_curvature_rpm_joint(pts1r, pts2, n_iter = 20, T_init = T_init, bend_init = bend_init, wsize = wsize, beta = beta, plotting = plotting, angle = angle, square_size = square_size1, circle_rad = circle_rad1)
    #plot_house(hc, x0sr, x1sr, x2sr, x3sr, x4sr, number_points)
    #plot_house(h, x0sr, x1sr, x2sr, x3sr, x4sr, number_points)
    #plot_box_circle(hc, square_size1, circle_rad1, angle, number_points = number_points)
    #plot_box_circle(h, square_size1, circle_rad1, angle, number_points = number_points)



    if 'I' in n:
        g = ThinPlateSpline(2)
        #plot_house(g, x0t, x1t, x2t, x3t, x4t, number_points)
        #plot_box_circle(g, square_size1, circle_rad1, 0, number_points = number_points)
        plot_grabs_two(g, pts2, 0)

    #plt.show()
    ipy.embed()

if __name__ == "__main__":
    main()



"""
xwarped = f.transform_points(Xs)
ewarped = f.transform_normals(Epts, Exs)
normalizations = nlg.norm(ewarped, axis = 1)
ewarped /= normalizations[:,None]
r = rads[i]
nw = normal_weights[i]
flipped_es = -ewarped


distmat = ssd.cdist(xwarped, Ys, 'sqeuclidean')
distmat_nn = ssd.cdist(ewarped, Eys, 'sqeuclidean')
distmat_fn = ssd.cdist(flipped_es, Eys, 'sqeuclidean')
distmat_n = np.minimum(distmat_nn, distmat_fn)

dist_xy = distmat/r + normal_coefs[i]*np.diagflat(indices).dot(distmat_n)/nw #normal_coefs[i]*np.diagflat(indices).dot(distmat_n)/nw
prob = np.exp(-dist_xy)

corr, _ , _ = balance_matrix3(prob, 20, x_priors, y_priors, outlierfrac)
corr += 1e-9
#ipy.embed()

wt_n = corr.sum(axis=1)

xtarg = (corr/wt_n).dot(Ys)
#etarg = tps_utils.find_all_normals_naive(xtarg, wsize = .7, orig_cloud = orig_target)
etarg = (corr/wt_n).dot(Eys)
#etarg /= nlg.norm(etarg, axis=1)[:,None]


wt_nn = np.r_[wt_n, wt_n]
wt_nn[n:] /= np.square(normalizations)
wt_nn /= 2*n
Exs_n = Exs.copy()
Exs_n *= normalizations[:,None]

f = fit_KrigingSpline(Xs, Epts, Exs_n, xtarg, etarg, bend_coef = regs[i], normal_coef = normal_coef, wt_n = wt_nn)



xwarped_nd = f.transform_points(x_nd)
corr_nm, r_n, _ = calc_correspondence_matrix(xwarped_nd, y_md, r=rads[i], p=.1, max_iter=10)
wt_n = corr_nm.sum(axis=1)

print i

targ_nd = (corr_nm/wt_n[:,None]).dot(y_md)


f = fit_ThinPlateSpline(x_nd, targ_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)

number_points = 50

bottom_row = np.c_[np.linspace(x0s[0], x1s[0], number_points), np.linspace(x0s[1], x1s[1], number_points)]
right_column = np.c_[np.linspace(x1s[0], x2s[0], number_points), np.linspace(x1s[1], x2s[1], number_points)]
right_diagonal = np.c_[np.linspace(x2s[0], x3s[0], number_points), np.linspace(x2s[1], x3s[1], number_points)]
left_diagonal = np.c_[np.linspace(x3s[0], x4s[0], number_points), np.linspace(x3s[1], x4s[1], number_points)]
left_column = np.c_[np.linspace(x4s[0], x0s[0], number_points), np.linspace(x4s[1], x0s[1], number_points)]

"""