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
from tn_rapprentice import tps
from tn_rapprentice import krig, tps as tn_tps
import tn_eval.tps_utils as tu
import numpy.linalg as nlg
from tn_rapprentice.tps import tps_eval, tps_grad, tps_fit3, tps_fit_regrot, tps_kernel_matrix, tps_cost
from tn_rapprentice import krig_utils as ku
from tn_eval import tps_utils
import IPython as ipy
import matplotlib.pyplot as plt
from tn_rapprentice import plotting_plt as p_plt


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


def fit_KrigingSpline_Interest(Xs, Epts, Exs, Ys, Eys, bend_coef = .1, normal_coef = 1, wt_n=None, alpha = 1.5, rot_coefs = 1e-5, interest_pts_inds = None):
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
    f.w_ng, f.trans_g, f.lin_ag = ku.krig_fit_interest(Xs, Ys, Epts, Exs, Eys, bend_coef = bend_coef, normal_coef = normal_coef, wt_n = wt_n, rot_coefs = rot_coefs, interest_pts_inds = interest_pts_inds)
    f.x_na, f.ex_na, f.exs = Xs, Epts, Exs
    return f

def fit_KrigingSpline(Xs, Epts, Exs, Ys, Eys, bend_coef = .01, normal_coef = 1, wt_n=None, alpha = 1.5, rot_coefs = 1e-5, interest_pts_inds = None):
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
    f.w_ng, f.trans_g, f.lin_ag = ku.krig_fit1Normal(1.5, Xs, Ys, Epts, Exs, Eys, bend_coef = bend_coef, normal_coef = normal_coef, wt_n = wt_n, rot_coefs = rot_coefs)
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
    f.w_ng, f.lin_ag = ku.krig_fit3_landmark(f.alpha, Xs, Ys, bend_coef,  wt_n = None)
    f.Xs = Xs
    return f

def fit_ThinPlateSpline_RotReg(x_na, y_ng, bend_coef = .1, rot_coefs = (0.01,0.01,0.0025),scale_coef=.01):
    import fastrapp
    f = ThinPlateSpline()
    rfunc = fastrapp.rot_reg
    fastrapp.set_coeffs(rot_coefs, scale_coef)
    f.lin_ag, f.trans_g, f.w_ng = tps.tps_fit_regrot(x_na, y_ng, bend_coef, rfunc)
    f.x_na = x_na
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
    #f = fit_KrigingSpline(x_nd, x_nd, x_nd, xtarg_nd, y_md, bend_coef = bend_coef, wt_n = wt_n, normal_coef = 0, rot_coefs = rot_reg)
    #f._bend_coef = bend_coef
    f._wt_n = wt_n
    f._rot_coef = rot_reg
    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, bend_coef, wt_n=wt_n)/wt_n.mean()
    return f

def fit_KrigingSpline_corr(x_nd, y_md, exs, eys, corr_nm, corr_nm_edge, bend_coef, rot_reg, normal_coef, normal_normalizers = None, x_weights = None):
    #Need to add Epts
    wt_n = corr_nm.sum(axis=1)
    wt_n_edge = corr_nm_edge.sum(axis=1)
    if np.any(wt_n == 0):
        inlier = wt_n != 0
        x_nd = x_nd[inlier,:]
        wt_n = wt_n[inlier,:]
        x_weights = x_weights[inlier]
        xtarg_nd = (corr_nm[inlier,:]/wt_n[:,None]).dot(y_md)
    elif np.any(wt_n_edge == 0):
        inlier = wt_n_edge != 0
        x_nd = x_nd[inlier,:]
        wt_n_edge = wt_n_edge[inlier,:]
        x_weights = x_weights[inlier]
        #xtarg_nd_edge = (corr_nm_edge[inlier,:]/wt_n_edge[:,None]).dot(y_md)
        targ_nd_edge = tps_utils.normal_corr_mult(corr_nm_edge/wt_n_edge[:,None], eys)
        targ_nd_edge /= nlg.norm(targ_nd_edge, axis=1)[:,None]
        xtarg_nd_edge = tps_utils.flip_normals(x_nd, x_nd, exs, targ_nd, targ_nd_edge)
    else:
        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        xtarg_nd_edge = (corr_nm_edge/wt_n_edge[:,None]).dot(eys)
    if x_weights is not None:
        if x_weights.ndim > 1:
            wt_n=wt_n[:,None]*x_weights
        else:
            wt_n=wt_n*x_weights
    if normal_coef != 0:
        if normal_normalizers is not None:

            wt_nn = np.r_[wt_n, ((1/normal_normalizers**2).T * wt_n_edge).T[:,0]]
            exs *= normal_normalizers
        else:
            wt_nn = np.r_[wt_n, wt_n_edge]
    else:
        wt_nn = wt_n

    #f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = bend_coef, wt_n = wt_n, rot_coef = rot_reg)
    #ipy.embed()
    f = fit_KrigingSpline(x_nd, x_nd, exs, xtarg_nd, xtarg_nd_edge, bend_coef = bend_coef, wt_n = wt_nn, normal_coef = normal_coef, rot_coefs = rot_reg)
    f._bend_coef = bend_coef
    f._wt_n = wt_n
    f._rot_coef = rot_reg
    #below could be buggy
    #f._cost = tps.krig_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, bend_coef, wt_n=wt_n)/wt_n.mean()
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
def tps_rpm_EM(x_nd, y_md,  n_iter=20, temp_init=.1,  temp_final=.01, bend_init=.1, bend_final=.01, rot_reg = 1e-5, 
             outlierfrac = 1e-2, EM_iter = 5, f_init = None, outlierprior = .1, plotting = False, angle = 0,
             square_size = 0, circle_rad = 0, wsize = .1):
    _,d=x_nd.shape
    temps = loglinspace(temp_init, temp_final, n_iter)
    bend_coefs = loglinspace(bend_init, bend_final, n_iter)
    if f_init is not None: 
        f = f_init  
    else:
        f = ThinPlateSpline(d)

    e1 = tps_utils.find_all_normals_naive(x_nd, wsize=wsize, flip_away=True)
    e2 = tps_utils.find_all_normals_naive(y_md, wsize=wsize, flip_away=True)

    for i in xrange(n_iter):
        for j in range(EM_iter):
            print i,j
            f, corr_nm = EM_step(f, x_nd, y_md, outlierfrac, temps[i], bend_coefs[i], rot_reg, outlierprior)
        #ipy.embed()
        if plotting and i%plotting==0:
            p_plt.plot_tps_registration(x_nd, y_md, f)
            #p_plt.plot_tps_registration_normals(x_nd, y_md, exs, eys, f, wsize = wsize)


    #ipy.embed()
    return f, corr_nm

#@profile




def tps_rpm_curvature_prior1(x_nd, y_md, orig_source = None, orig_target = None, n_iter=20, temp_init=.1,  temp_final=.01, bend_init=.1, bend_final=.01,
                     rot_reg = 1e-5,  outlierfrac = 1e-2, wsize = .1, EM_iter = 5, f_init = None, outlierprior = .1, beta = 1, plotting = False, angle = 0,
                     square_size = 0, circle_rad = 0):
    _,d=x_nd.shape
    temps = loglinspace(temp_init, temp_final, n_iter)
    bend_coefs = loglinspace(bend_init, bend_final, n_iter)
    if f_init is not None: 
        f = f_init  
    else:
        f = ThinPlateSpline(d)
        # f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)

    if orig_source is None:
        orig_source = x_nd
    if orig_target is None:
        orig_target = y_md

    curve_cost = tps_utils.compute_curvature_weights(x_nd, y_md, wsize = wsize)

    for i in xrange(n_iter):
        for j in range(EM_iter):
            print i, j
            f, corr_nm = EM_step(f, x_nd, y_md, outlierfrac, temps[i], bend_coefs[i], rot_reg, outlierprior, curve_cost = curve_cost, beta = beta)
        #ipy.embed()
        if plotting and i%plotting==0:
            p_plt.plot_tps_registration(x_nd, y_md, f)
            #p_plt.plot_tps_registration_normals(x_nd, y_md, e1, e2, f, wsize = .1)
  
            
    #ipy.embed()
    return f, corr_nm


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
      
    f = fit_KrigingSpline_corr(x_nd, y_md, x_nd, y_md, corr_nm, corr_nm, bend_coef, rot_reg, normal_coef = 0)
    #p_plt.plot_tps_registration(x_nd, y_md, f)
    
    return f, corr_nm

##############################################################
##############################################################
##############################################################
##############################################################
# IMPORTANT STUFF BELOW!!!
def tps_n_rpm_final_hopefully(x_nd, y_md, exs = None, eys = None, orig_source = None, orig_target = None, n_iter=20, temp_init=.1,  temp_final=.01, bend_init=.1, bend_final=.01,
                     rot_reg = 1e-5,  outlierfrac = 1e-2, wsize = .1, EM_iter = 5, f_init = None, outlierprior = .1, beta = 1, plotting = False, jplotting = 0, 
                    normal_coef = .1,  normal_temp = .05, flip_away=True):
    
    _,d=x_nd.shape
    temps = loglinspace(temp_init, temp_final, n_iter)
    bend_coefs = loglinspace(bend_init, bend_final, n_iter)
    normal_temp = normal_temp
    #ipy.embed()
    
    if f_init is not None: 
        f = f_init  
    else:
        f = fit_KrigingSpline(x_nd, x_nd, x_nd, x_nd, x_nd, normal_coef = 0)
        # f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)
    if orig_source is None:
        orig_source = x_nd
    if orig_target is None:
        orig_target = y_md
    
    if exs is None:
        exs = tps_utils.find_all_normals_naive(x_nd, wsize=wsize, flip_away=flip_away)
    if eys is None:
        eys = tps_utils.find_all_normals_naive(y_md, wsize=wsize, flip_away=flip_away)

    curve_cost = tps_utils.compute_curvature_weights(x_nd, y_md, wsize = wsize)
    normal_temp = normal_temp
    
    for i in xrange(n_iter):
        if i == n_iter - 1:
            #print "--------------------------------------"
            for j in range(EM_iter):
                f, corr_nm, corr_nm_edge = EM_step_final(f, x_nd, y_md, exs, eys, outlierfrac, temps[i], normal_temp, bend_coefs[i], normal_coef, rot_reg, outlierprior, beta = beta, wsize = wsize)
                #print tps_n_rpm_obj(f, corr_nm, corr_nm_edge, temps[i], bend_coefs[i], normal_temp, x_nd, y_md, exs, eys, normal_coef)
        else:
            #print "--------------------------------------"
            for j in range(EM_iter):
                f, corr_nm = EM_step(f, x_nd, y_md, outlierfrac, temps[i], bend_coefs[i], rot_reg, outlierprior, curve_cost = curve_cost, beta = beta)
                #print tps_rpm_obj(f, corr_nm, temps[i], bend_coefs[i],  x_nd, y_md)

        if plotting and i%plotting==0:
            p_plt.plot_tps_registration(x_nd, y_md, f)
            p_plt.plot_tps_registration_normals(x_nd, y_md, exs, eys, f, wsize = wsize)
            targ_nd = (corr_nm/np.sum(corr_nm, axis=1)[:,None]).dot(y_md)
            #targ_nd_edge = tps_utils.find_all_normals_naive(y_md, wsize = wsize)
            if i == n_iter - 1:
                wt_n_edge = corr_nm_edge.sum(axis=1)
                targ_nd_edge = (corr_nm_edge/wt_n_edge[:,None]).dot(eys)
                targ_nd_edge = tps_utils.normal_corr_mult(corr_nm_edge/wt_n_edge[:,None], eys)
                targ_nd_edge = tps_utils.flip_normals(x_nd, x_nd, exs, targ_nd, targ_nd_edge)
                p_plt.plot_corr_normals(x_nd, exs, targ_nd, targ_nd_edge)
    
    return f, corr_nm, corr_nm_edge
    
def EM_step_final(f, x_nd, y_md, exs, eys, outlierfrac, temp, bend_coef, normal_temp, normal_coef, rot_reg, outlierprior, curve_cost = None, beta = 1, wsize = .1):
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

    
    
    ewarped_nd = f.transform_normals(x_nd, exs)
    ewarped_nd = tps_utils.flip_normals(x_nd, x_nd, exs, xwarped_nd, ewarped_nd)
    Alphas = nlg.norm(ewarped_nd, axis=1)[:,None]
    ewarped_nd /= Alphas
    dist_nm_edge_warped = ssd.cdist(ewarped_nd, eys, 'sqeuclidean')
    ewarped_nd_flipped = -ewarped_nd
    dist_nm_edge_flipped = ssd.cdist(ewarped_nd_flipped, eys, 'sqeuclidean')
    dist_nm_edge = np.minimum(dist_nm_edge_warped, dist_nm_edge_flipped)

    nt = normal_temp
    #prob_nm_edge = np.exp(-dist_nm_edge_warped/nt)
    prob_nm_edge = np.exp(-dist_nm_edge/nt)

    corr_nm_edge, r_N_edge, _ = balance_matrix3(prob_nm_edge, 10, x_priors, y_priors, outlierfrac)
    corr_nm_edge += 1e-9
    curve_weights = tps_utils.compute_curvature_weights(x_nd, y_md, wsize = wsize)
    corr_nm_edge *= curve_weights #gives normals weights based on curvature. The higher the curvature, the less weight it gets

    wt_n_edge = corr_nm_edge.sum(axis=1)    

    targ_nd_edge = (corr_nm_edge/wt_n_edge[:,None]).dot(eys)
    targ_nd_edge = tps_utils.normal_corr_mult(corr_nm_edge/wt_n_edge[:,None], eys)
    targ_nd_edge = tps_utils.flip_normals(x_nd, x_nd, exs, targ_nd, targ_nd_edge)
    #targ_nd_edge = tps_utils.find_all_normals_naive(targ_nd, wsize = .15)

    f = fit_KrigingSpline_final(x_nd, y_md, exs, eys, corr_nm, corr_nm_edge, bend_coef, rot_reg, normal_coef)

    p_plt.plot_tps_registration(x_nd, y_md, f)
    p_plt.plot_tps_registration_normals(x_nd, y_md, exs, eys, f, wsize = wsize)
    p_plt.plot_corr_normals(x_nd, exs, targ_nd, targ_nd_edge)

    return f, corr_nm, corr_nm_edge
    #ipy.embed()

def fit_KrigingSpline_final(x_nd, y_md, exs, eys, corr_nm, corr_nm_edge,bend_coef, rot_reg, normal_coef, x_weights = None):
    wt_n = corr_nm.sum(axis=1)
    wt_n_edge = corr_nm_edge.sum(axis=1)    
    
    if np.any(wt_n == 0):
        inlier = wt_n != 0
        x_nd = x_nd[inlier,:]
        wt_n = wt_n[inlier]
        x_weights = x_weights[inlier]
        xtarg_nd = (corr_nm[inlier,:]/wt_n[:,None]).dot(y_md)
    else:
        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
    if normal_coef != 0:
        inlier = wt_n_edge != 0
        exs = exs[inlier,:]
        #ipy.embed()
        wt_n_edge = wt_n_edge[inlier]
        #targ_nd_edge = (corr_nm_edge/wt_n_edge[:,None]).dot(eys)
        targ_nd_edge = tps_utils.normal_corr_mult(corr_nm_edge[inlier,:]/wt_n_edge[:,None], eys)
        targ_nd_edge = tps_utils.flip_normals(x_nd, x_nd, exs, xtarg_nd, targ_nd_edge)
        #targ_nd_edge = tps_utils.find_all_normals_naive(targ_nd, wsize = .15)
    
    if x_weights is not None:
        if x_weights.ndim > 1:
            wt_n=wt_n[:,None]*x_weights
        else:
            wt_n=wt_n*x_weights

    else:
        if normal_coef != 0:
            wt_nn = np.r_[wt_n, wt_n_edge]
        else:
            wt_nn = wt_n

    #f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = bend_coef, wt_n = wt_n, rot_coef = rot_reg)
    #ipy.embed()
    if normal_coef != 0:
        f = fit_KrigingSpline(x_nd, x_nd, exs, xtarg_nd, targ_nd_edge, bend_coef = bend_coef, wt_n = wt_nn, normal_coef = normal_coef, rot_coefs = rot_reg)
    else:
        f = fit_KrigingSpline(x_nd, x_nd, x_nd, xtarg_nd, x_nd, bend_coef = bend_coef, wt_n = wt_nn, normal_coef = normal_coef, rot_coefs = rot_reg)
    f._bend_coef = bend_coef
    f._wt_n = wt_n
    f._rot_coef = rot_reg
    #below could be buggy
    #f._cost = tps.krig_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, bend_coef, wt_n=wt_n)/wt_n.mean()
    return f
##############################################################
##############################################################
##############################################################
##############################################################
# IMPORTANT STUFF ABOVE!!!

def tps_n_rpm_obj(f, corr, corr_edge, temp, bend_coef, normal_temp, x_nd, y_md, exs, eys, normal_coef):
    fpoints = f.transform_points(x_nd)
    fnormals = f.transform_normals(x_nd, exs)
    point_dist = ssd.cdist(fpoints, y_md, 'sqeuclidean')
    normals_dist = ssd.cdist(fnormals, eys, 'sqeuclidean')
    point_dist *= corr
    normals_dist *= corr_edge*normal_coef
    
    obj = 0
    obj += np.sum(point_dist)
    obj += np.sum(normals_dist)

    S = ku.krig_kernel_mat(1.5, f.x_na, f.ex_na, f.exs)
    D = ku.krig_mat_linear(f.x_na, f.ex_na, f.exs)
    B = ku.bending_energy_mat(S, D)

    cost = 0
    for w in f.w_ng.T:
        cost += w.dot(B.dot(w))
    obj += cost
    obj += temp*np.sum(corr*np.log(corr))
    obj += normal_temp*np.sum(corr_edge*np.log(corr_edge))
    #ipy.embed()

    return obj

def tps_rpm_obj(f, corr, temp, bend_coef,  x_nd, y_md):
    n = len(x_nd)
    fpoints = f.transform_points(x_nd)
    point_dist = ssd.cdist(fpoints, y_md, 'sqeuclidean')
    point_dist *= corr

    obj = 0
    obj += np.sum(point_dist)

    S = ku.krig_mat_landmark(1.5, x_nd)
    D = np.c_[np.ones((n,1)), x_nd]
    B = ku.bending_energy_mat(S,D)
    
    cost = 0
    for w in f.w_ng.T:
        cost += w.dot(B.dot(w))
    obj += cost
    obj += temp*np.sum(corr*np.log(corr))

    #ipy.embed()
    return obj

def close_windows():
    for i in range(20):
        plt.close()

def main():
    from tn_testing.test_tps import gen_half_sphere, gen_half_sphere_pulled_in, gen_house, gen_box_circle, gen_circle_points
    from tn_eval.tps_utils import find_all_normals_naive
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from tn_testing import grabs_two
    from tn_rapprentice.plotting_plt import plot_house, plot_box_circle
    from big_rope_pcloud import old_cloud, new_cloud



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
    # ipy.embed()
    plt.ion()

    number_points = 30
    square_size1 = 1
    square_size2 = 1
    circle_rad1 = .5
    circle_rad2 = .5

    angle = 0
    

    x0s, x1s, x2s, x3s, x4s = np.array([0,0]), np.array([1,0]), np.array([1,1]), np.array([.5, 3]), np.array([0,1])
    x0t, x1t, x2t, x3t, x4t = np.array([0,0]), np.array([1,0]), np.array([1,.5]), np.array([.5, 1.5]), np.array([0,.5])

    #pts1 = gen_house(x0s, x1s, x2s, x3s, x4s, number_points)
    #pts2 = gen_house(x0t, x1t, x2t, x3t, x4t, number_points)    
    #pts1 = gen_box_circle(square_size1, circle_rad1, number_points = number_points)
    #pts2 = gen_box_circle(square_size2, circle_rad2, number_points = number_points)
    #pts1 = gen_circle_points(.5, 30)
    #pts2 = gen_circle_points(.5, 30)
    pts1 = grabs_two.old_cloud[:,:2]
    pts2 = grabs_two.new_cloud[:,:2]


    
    #pts1r = np.random.permutation(pts1)
    pts1r = rotate_point_cloud2d(pts1, angle)
    x0sr, x1sr, x2sr, x3sr, x4sr = rotate_point2d(x0s, angle),rotate_point2d(x1s, angle),rotate_point2d(x2s, angle),rotate_point2d(x3s, angle),rotate_point2d(x4s, angle)


    EM_iter = 5

    beta = .01 #20 works for 90 rotation
    wsize = .15
    jplotting = 0
    plotting = 1
    


    temp_init = 1
    temp_final = .0005
    bend_init = 1e2 #1e2 works for 90 rotation
    bend_final = 1e-1
    
    normal_coef = 100
    normal_temp = .001

    exs = find_all_normals_naive(pts1r, wsize = wsize, flip_away=True)
    eys = find_all_normals_naive(pts2, wsize = wsize, flip_away=True)

    #ipy.embed()
    #ipy.embed()

    n = 4 , #'I'

    if 1 in n:
        f1 , corr1 = tps_rpm_curvature_prior1(pts1r, pts2,  n_iter = 20, EM_iter = EM_iter, temp_init = temp_init, temp_final = temp_final, bend_init = bend_init, bend_final = bend_final, wsize = wsize, beta = beta, plotting = plotting, angle = angle, square_size = square_size1, circle_rad = circle_rad1)
        #plot_house(f1, x0sr, x1sr, x2sr, x3sr, x4sr, number_points)
        #plot_box_circle(f1, square_size1, circle_rad1, angle, number_points = number_points)
        #plot_grabs_two(f1, pts1r, angle)
        p_plt.plot_tps_registration(pts1r, pts2, f1)

    if 3 in n:
        f3, corr3 = tps_rpm_bij(pts1r,pts2, reg_init = temp_init, rad_init = bend_init)
        plot_house(f3, x0sr, x1sr, x2sr, x3sr, x4sr, number_points)
        #plot_box_circle(f3, square_size1, circle_rad1, angle, number_points = number_points)
        #plot_grabs_two(f3, pts1r, angle)

    if 4 in n:
        f4,corr4 = tps_rpm_EM(pts1r, pts2, n_iter = 20, EM_iter = EM_iter, temp_init = temp_init, temp_final = temp_final, bend_init = bend_init, bend_final = bend_final, plotting = plotting, angle = angle, square_size = square_size1, circle_rad = circle_rad1, wsize = wsize)
        #plot_house(f4, x0sr, x1sr, x2sr, x3sr, x4sr, number_points)
        #plot_box_circle(f4, square_size1, circle_rad1, angle, number_points = number_points)
        p_plt.plot_tps_registration(pts1r, pts2, f4)

    if 8 in n:
        f8, corr8, corr_edge8 = tps_n_rpm_final_hopefully(pts1r, pts2, n_iter = 20, EM_iter = EM_iter, temp_init = temp_init, temp_final=temp_final, normal_temp = normal_temp, bend_init = bend_init, bend_final = bend_final, normal_coef = normal_coef, plotting = plotting, jplotting = jplotting, beta = beta, wsize = wsize)
        #plot_house(f8, x0sr, x1sr, x2sr, x3sr, x4sr,  number_points) 
        #plot_box_circle(f8, square_size1, circle_rad1, angle, number_points = number_points)
        #plot_grabs_two(f6, pts1r, angle)
        p_plt.plot_tps_registration(pts1r, pts2, f8)



    if 'I' in n:
        g = ThinPlateSpline(2)
        #plot_house(g, x0t, x1t, x2t, x3t, x4t, number_points)
        plot_box_circle(g, square_size1, circle_rad1, 0, number_points = number_points)
        #plot_grabs_two(g, pts2, 0)

    plt.show()
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