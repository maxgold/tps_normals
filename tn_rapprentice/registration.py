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
import tn_eval.tps_utils as tu
import numpy.linalg as nlg
from tn_rapprentice.tps import tps_eval, tps_grad, tps_fit3, tps_fit_regrot, tps_kernel_matrix, tps_cost
# from svds import svds
from tn_rapprentice import krig_utils as ku


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

def fit_KrigingSplineWeird(Xs, Epts, Exs, Ys, Eys, bend_coef = .1, alpha = 1.5, normal_coef = 1, wt_n=None):
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
    f.w_ng, f.trans_g, f.lin_ag = ku.krig_fit1Weird(f.alpha, Xs, Ys, Epts, Exs, Eys, bend_coef = bend_coef, normal_coef = normal_coef, wt_n = wt_n)
    f.x_na, f.ex_na, f.exs = Xs, Epts, Exs
    return f

def fit_KrigingSpline(Xs, Epts, Exs, Ys, Eys, bend_coef = 1e-6, normal_coef = 1, wt_n=None, alpha = 1.5, rot_coefs = 1e-5):
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
    f.w_ng, f.trans_g, f.lin_ag = ku.krig_fit1Normal(f.alpha, Xs, Ys, Epts, Exs, Eys, bend_coef = bend_coef, normal_coef = normal_coef, wt_n = wt_n, rot_coefs = rot_coefs)
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
    


def unit_boxify(x_na):    
    ranges = x_na.ptp(axis=0)
    dlarge = ranges.argmax()
    unscaled_translation = - (x_na.min(axis=0) + x_na.max(axis=0))/2
    scaling = 1./ranges[dlarge]
    scaled_translation = unscaled_translation * scaling
    return x_na*scaling + scaled_translation, (scaling, scaled_translation)
    
def unscale_tps_3d(f, src_params, targ_params):
    """Only works in 3d!!"""
    assert len(f.trans_g) == 3
    p,q = src_params
    r,s = targ_params
    print p,q,r,s
    fnew = ThinPlateSpline()
    fnew.x_na = (f.x_na  - q[None,:])/p 
    fnew.w_ng = f.w_ng * p / r
    fnew.lin_ag = f.lin_ag * p / r
    fnew.trans_g = (f.trans_g  + f.lin_ag.T.dot(q) - s)/r
    
    return fnew

def unscale_tps(f, src_params, targ_params):
    """Only works in 3d!!"""
    p,q = src_params
    r,s = targ_params
    
    d = len(q)
    
    lin_in = np.eye(d)*p
    trans_in = q
    aff_in = Affine(lin_in, trans_in)
    
    lin_out = np.eye(d)/r
    trans_out = -s/r
    aff_out = Affine(lin_out, trans_out)

    return Composition([aff_in, f, aff_out])
    
    

def tps_rpm(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg=1e-4,
            plotting = False, f_init = None, plot_cb = None):
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
        # f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)

    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        corr_nm, r_n, _ = calc_correspondence_matrix(xwarped_nd, y_md, r=rads[i], p=.1, max_iter=10)

        wt_n = corr_nm.sum(axis=1)


        targ_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, targ_nd, corr_nm, wt_n, f)
        
        
        f = fit_ThinPlateSpline(x_nd, targ_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)

    return f


#adjust to account for Epts
def tps_rpm_normals(x_nd, y_md, exs, eys,  n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, normal_weight = 1, alpha = 1.5, 
            point_init = 1, point_final = 1, normal_init = 1, normal_final = .1, normal_coef = 1, rot_reg = 1e-5):
    #the problem is that some entries become outliers 

    n,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    norms = loglinspace(normal_init, normal_final, n_iter)
    points = loglinspace(point_init, point_final, n_iter)


    f = fit_KrigingSpline(x_nd, x_nd, exs, x_nd, exs)

    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        ewarped_xs = f.transform_normals(x_nd, exs)
        r = rads[i]

        distmat = ssd.cdist(xwarped_nd,y_md, 'euclidean')
        distmat_n = ssd.cdist(ewarped_xs, eys, 'cosine')
        #dist_nm = points[i]*distmat[interest_pts_inds] + norms[i]*distmat_n
        dist_nm = distmat + 10*distmat_n

        prob_nm = np.exp(-dist_nm/(2*r))

        corr_nm =  balance_matrix(prob_nm, .1)
        corr_nm += 1e-9

        wt_n = corr_nm.sum(axis=1)
        #wt_ni = corr_nm[interest_pts_inds].sum(axis=1)
        wt_nn = np.r_[wt_n, wt_n]

        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        etarg_ys = (corr_nm/wt_n[:,None]).dot(eys) #have to normalize this
        etarg_ys = etarg_ys/nlg.norm(etarg_ys, axis=1)[:,None]

        f = fit_KrigingSpline(x_nd, x_nd, exs, xtarg_nd, etarg_ys, bend_coef = regs[i], wt_n=wt_nn, normal_coef=normal_coef) #vary normal coefficient? Maybe set it to norms[i] then put another fit outside loop with normal_coef = 1 or whatever
        f._corr = corr_nm
        f._bend_coef = regs[i]
        f._rot_coef = rot_reg
        f._wt_n = wt_n
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

        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, outlierfrac) # edit final value to change outlier percentage
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
    return f,g

def tps_rpm_bij_normals(x_nd, y_md, exs, eys, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3, 
            plotting = False, plot_cb = None, x_weights = None, y_weights = None, outlierprior = .1, outlierfrac = 2e-1, vis_cost_xy = None,
            point_init = .1, point_final = 1, normal_init = 1, normal_final = .1, normal_coef = 1):
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
    norms = loglinspace(normal_init, normal_final, n_iter)
    points = loglinspace(point_init, point_final, n_iter)

    f = fit_KrigingSpline(x_nd, x_nd, exs, x_nd, exs)
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0) # align the medians
    # do a coarse search through rotations
    # fit_rotation(f, x_nd, y_md)
    
    g = fit_KrigingSpline(y_md, y_md, eys, y_md, eys)
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
        exwarped_nd = f.transform_normals(x_nd, exs)
        eywarped_md = g.transform_normals(y_md, eys)
        
        fwddist_nm = ssd.cdist(xwarped_nd, y_md,'euclidean')
        invdist_nm = ssd.cdist(x_nd, ywarped_md,'euclidean')
        fw_norm_dist = ssd.cdist(exwarped_nd, eys, 'cosine')
        inv_norm_dist = ssd.cdist(exs, eywarped_md, 'cosine')
        
        r = rads[i]
        p = points[i]
        n = norms[i]
        prob_nm = np.exp(-(p*(fwddist_nm + invdist_nm) + n*(fw_norm_dist + inv_norm_dist))/ (4*r) )
        """
        if vis_cost_xy != None:
            pi = np.exp( -vis_cost_xy )
            pi /= pi.max() # rescale the maximum probability to be 1. effectively, the outlier priors are multiplied by a visual prior of 1 (since the outlier points have a visual prior of 1 with any point)
            prob_nm *= pi
        """

        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, outlierfrac) # edit final value to change outlier percentage
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)

        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)
        extarg_nd = (corr_nm/wt_n[:,None]).dot(eys)
        eytarg_md = (corr_nm/wt_m[None,:]).T.dot(exs)
        extarg_nd = extarg_nd/nlg.norm(extarg_nd, axis=1)[:,None]
        eytarg_md = eytarg_md/nlg.norm(eytarg_md, axis=1)[:,None]
        
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f)
        
        if i == (n_iter-1):
            if x_weights is not None:
                wt_n=wt_n*x_weights
            if y_weights is not None:
                wt_m=wt_m*y_weights
        wt_nn = np.r_[wt_n, wt_n]
        wt_mm = np.r_[wt_m, wt_m]
        f = fit_KrigingSpline(x_nd, x_nd, exs, xtarg_nd, extarg_nd, bend_coef = regs[i], wt_n=wt_nn, rot_coefs = rot_reg, normal_coef = normal_coef)
        g = fit_KrigingSpline(y_md, y_md, eys, ytarg_md, eytarg_md, bend_coef = regs[i], wt_n=wt_mm, rot_coefs = rot_reg, normal_coef = normal_coef)
        
        # add metadata of the transformation f
        f._corr = corr_nm
        f._bend_coef = regs[i]
        f._rot_coef = rot_reg
        f._wt_n = wt_nn
    
    f._cost = krig_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, f.exs, xtarg_nd, extarg_nd, regs[i], wt_n=wt_nn)/wt_n.mean()
    g._cost = krig_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, g.exs, ytarg_md, eytarg_md, regs[i], wt_n=wt_mm)/wt_m.mean()
    return f,g




def tps_reg_cost(f):
    K_nn = tps.tps_kernel_matrix(f.x_na)
    cost = 0
    for w in f.w_ng.T:
        cost += w.dot(K_nn.dot(w))
    return cost

def krig_reg_cost(f):
    K_nn = ku.krig_kernel_mat(1.5, f.x_na, f.ex_na, f.exs)
    cost = 0
    for w in f.w_ng.T:
        cost += w.dot(K_nn.dot(w))
    return cost
    
def logmap(m):
    "http://en.wikipedia.org/wiki/Axis_angle#Log_map_from_SO.283.29_to_so.283.29"
    theta = np.arccos(np.clip((np.trace(m) - 1)/2,-1,1))
    return (1/(2*np.sin(theta))) * np.array([[m[2,1] - m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1]]]), theta


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
def balance_matrix33(prob_nm, max_iter, p, outlierfrac, r_N = None):
    
    n,m = prob_nm.shape
    prob_NM = np.empty((n+1, m+1), 'f4')
    prob_NM[:n, :m] = prob_nm
    prob_NM[:n, m] = p
    prob_NM[n, :m] = p
    prob_NM[n, m] = p*np.sqrt(n*m)
    
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

def calc_correspondence_matrix_normals(f, Xs, Ys, Eys, p, outlierfrac):
    distmat = ssd.cdist(Xs,Ys)
    distmat_n = ssd.cdist(f.transform_normals(Xs), Eys)
    dist_nm = distmat + distmat_n

    prob_nm = np.exp(-dist_nm/r)

    outlierfrac = .1
    return banlance_matrix3(prob_nm, max_iter, p, outlierfrac)


def nan2zero(x):
    np.putmask(x, np.isnan(x), 0)
    return x


def fit_score(src, targ, dist_param):
    "how good of a partial match is src to targ"
    sqdists = ssd.cdist(src, targ,'sqeuclidean')
    return -np.exp(-sqdists/dist_param**2).sum()

def orthogonalize3_cross(mats_n33):
    "turns each matrix into a rotation"

    x_n3 = mats_n33[:,:,0]
    # y_n3 = mats_n33[:,:,1]
    z_n3 = mats_n33[:,:,2]

    znew_n3 = math_utils.normr(z_n3)
    ynew_n3 = math_utils.normr(np.cross(znew_n3, x_n3))
    xnew_n3 = math_utils.normr(np.cross(ynew_n3, znew_n3))

    return np.concatenate([xnew_n3[:,:,None], ynew_n3[:,:,None], znew_n3[:,:,None]],2)

def orthogonalize3_svd(x_k33):
    u_k33, _s_k3, v_k33 = svds.svds(x_k33)
    return (u_k33[:,:,:,None] * v_k33[:,None,:,:]).sum(axis=2)

def orthogonalize3_qr(_x_k33):
    raise NotImplementedError

def krig_cost(lin_ag, trans_g, w_ng, x_na, y_ng, exs, eys, bend_coef, K_nn = None, return_tuple = False, wt_n = None):
    d = lin_ag.shape[0]
    if K_nn is None: K_nn = ku.krig_kernel_mat(1.5, x_na, x_na, exs)
    D = ku.krig_mat_linear(x_na, x_na, exs)
    if wt_n is None: wt_n = np.ones(len(x_na))
    ypred_ng = np.dot(K_nn, w_ng) + np.dot(D[:, 1:], lin_ag) + trans_g[None, :]
    res_cost = (wt_n[:,None]*(ypred_ng-np.r_[y_ng, eys])**2).sum()
    bend_cost = bend_coef*sum(np.dot(w_ng[:,g], np.dot(K_nn, w_ng[:,g])) for g in xrange(d))
    if return_tuple:
        return res_cost, bend_cost, res_cost + bend_cost
    else:
        return res_cost + bend_cost

def tps_rpm_bij_normals_naive1(x_nd, y_md, exs, eys, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3, 
            plotting = False, plot_cb = None, x_weights = None, y_weights = None, outlierprior = .1, outlierfrac = 2e-1, vis_cost_xy = None, normal_coef = 1):
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

    f = fit_KrigingSpline(x_nd, x_nd, exs, x_nd, exs)
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0) # align the medians
    # do a coarse search through rotations
    # fit_rotation(f, x_nd, y_md)
    
    g = fit_KrigingSpline(y_md, y_md, eys, y_md, eys)
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

        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, outlierfrac) # edit final value to change outlier percentage
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)

        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)
        extarg_nd = (corr_nm/wt_n[:,None]).dot(eys)
        eytarg_md = (corr_nm/wt_m[None,:]).T.dot(exs)
        
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f)
        
        if i == (n_iter-1):
            if x_weights is not None:
                wt_n=wt_n*x_weights
            if y_weights is not None:
                wt_m=wt_m*y_weights
        f = fit_KrigingSpline(x_nd, x_nd, exs, xtarg_nd, extarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coefs = rot_reg, normal_coef = 0)
        g = fit_KrigingSpline(y_md, y_md, eys, ytarg_md, eytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coefs = rot_reg, normal_coef = 0)
        
        # add metadata of the transformation f
        f._corr = corr_nm
        f._bend_coef = regs[i]
        f._rot_coef = rot_reg
        f._wt_n = wt_n
    
    f = fit_KrigingSpline(x_nd, x_nd, exs, xtarg_nd, extarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coefs = rot_reg)
    g = fit_KrigingSpline(y_md, y_md, eys, ytarg_md, eytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coefs = rot_reg)
    
    f._cost = tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, regs[i], wt_n=wt_n)/wt_n.mean()
    g._cost = tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ytarg_md, regs[i], wt_n=wt_m)/wt_m.mean()
    return f,g

def tps_rpm_bij_normals_naive2(x_nd, y_md, exs, eys, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3, 
            plotting = False, plot_cb = None, x_weights = None, y_weights = None, outlierprior = .1, outlierfrac = 2e-1, vis_cost_xy = None, normal_coef = 1):
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

    f = fit_KrigingSpline(x_nd, x_nd, exs, x_nd, exs)
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0) # align the medians
    # do a coarse search through rotations
    # fit_rotation(f, x_nd, y_md)
    
    g = fit_KrigingSpline(y_md, y_md, eys, y_md, eys)
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

        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, outlierfrac) # edit final value to change outlier percentage
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)

        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)
        extarg_nd = (corr_nm/wt_n[:,None]).dot(eys)
        eytarg_md = (corr_nm/wt_m[None,:]).T.dot(exs)
        
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f)
        
        if i == (n_iter-1):
            if x_weights is not None:
                wt_n=wt_n*x_weights
            if y_weights is not None:
                wt_m=wt_m*y_weights
        f = fit_KrigingSpline(x_nd, x_nd, exs, xtarg_nd, extarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coefs = rot_reg, normal_coef = normal_coef)
        g = fit_KrigingSpline(y_md, y_md, eys, ytarg_md, eytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coefs = rot_reg, normal_coef = normal_coef)
        
        # add metadata of the transformation f
        f._corr = corr_nm
        f._bend_coef = regs[i]
        f._rot_coef = rot_reg
        f._wt_n = wt_n
    
    f._cost = tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, regs[i], wt_n=wt_n)/wt_n.mean()
    g._cost = tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ytarg_md, regs[i], wt_n=wt_m)/wt_m.mean()
    return f,g

def tps_rpm_normals_naive1(x_nd, y_md, exs, eys, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg=1e-4,
            plotting = False, f_init = None, plot_cb = None, normal_coef = 1):
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
        f = fit_KrigingSpline(x_nd, x_nd, exs, x_nd, exs)
        # f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)

    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        corr_nm, r_n, _ = calc_correspondence_matrix(xwarped_nd, y_md, r=rads[i], p=.1, max_iter=10)

        wt_n = corr_nm.sum(axis=1)
        wt_nn = np.r_[wt_n, wt_n]


        targ_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        etarg_nd = (corr_nm/wt_n[:,None]).dot(eys)
        
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, targ_nd, corr_nm, wt_n, f)
        
        
        f = fit_KrigingSpline(x_nd, x_nd, exs, targ_nd, etarg_nd, bend_coef = regs[i], wt_n=wt_nn, rot_coefs = rot_reg, normal_coef = 0)

    f = fit_KrigingSpline(x_nd, x_nd, exs, targ_nd, etarg_nd, bend_coef = regs[i], wt_n=wt_nn, rot_coefs = rot_reg, normal_coef=normal_coef)
    f._corr = corr_nm
    f._bend_coef = regs[i]
    f._rot_coef = rot_reg
    f._wt_n = wt_n

    return f, corr_nm

def tps_rpm_normals_naive2(x_nd, y_md, exs, eys, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg=1e-4,
            plotting = False, f_init = None, plot_cb = None, normal_coef = 1):
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
        f = fit_KrigingSpline(x_nd, x_nd, exs, x_nd, exs)
        # f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)

    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        corr_nm, r_n, _ = calc_correspondence_matrix(xwarped_nd, y_md, r=rads[i], p=.1, max_iter=10)

        wt_n = corr_nm.sum(axis=1)
        wt_nn = np.r_[wt_n, wt_n]


        targ_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        etarg_nd = (corr_nm/wt_n[:,None]).dot(eys)
        
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, targ_nd, corr_nm, wt_n, f)
        
        
        f = fit_KrigingSpline(x_nd, x_nd, exs, targ_nd, etarg_nd, bend_coef = regs[i], wt_n=wt_nn, rot_coefs = rot_reg, normal_coef=normal_coef)
    f._corr = corr_nm
    f._bend_coef = regs[i]
    f._rot_coef = rot_reg
    f._wt_n = wt_n

    return f, corr_nm



def tps_rpm_normals_interest(x_nd, y_md, exs, eys,  Epts = None, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, normal_weight = 1, alpha = 1.5, 
            point_init = 1, point_final = 1, normal_init = 1, normal_final = 1, normal_coef = 1, rot_reg = 1e-5):
    #the problem is that some entries become outliers 

    n,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    norms = loglinspace(normal_init, normal_final, n_iter)
    points = loglinspace(point_init, point_final, n_iter)


    f = fit_KrigingSpline(x_nd, Epts, exs, x_nd, exs)

    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        ewarped_xs = f.transform_normals(Epts, exs)
        r = rads[i]

        distmat = ssd.cdist(xwarped_nd,y_md, 'euclidean')
        distmat_n = ssd.cdist(ewarped_xs, eys, 'cosine')
        #dist_nm = points[i]*distmat[interest_pts_inds] + norms[i]*distmat_n
        dist_nm = distmat + normal_final*distmat_n

        prob_nm = np.exp(-dist_nm/(2*r))

        corr_nm =  balance_matrix(prob_nm, .1)
        corr_nm += 1e-9

        wt_n = corr_nm.sum(axis=1)
        #wt_ni = corr_nm[interest_pts_inds].sum(axis=1)
        wt_nn = np.r_[wt_n, wt_n]

        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        etarg_ys = (corr_nm/wt_n[:,None]).dot(eys) #have to normalize this
        etarg_ys = etarg_ys/nlg.norm(etarg_ys, axis=1)[:,None]

        f = fit_KrigingSpline(x_nd, Epts, exs, xtarg_nd, etarg_ys, bend_coef = regs[i], wt_n=wt_nn, normal_coef=normal_coef) #vary normal coefficient? Maybe set it to norms[i] then put another fit outside loop with normal_coef = 1 or whatever
        f._corr = corr_nm
        f._bend_coef = regs[i]
        f._rot_coef = rot_reg
        f._wt_n = wt_n
    return f, corr_nm




def main():
    from tn_testing.test_tps import gen_half_sphere, gen_half_sphere_pulled_in
    from tn_eval.tps_utils import find_all_normals_naive
    pts1 = gen_half_sphere(1, 30)
    pts2 = gen_half_sphere_pulled_in(1, 30, 4, .2)
    e1 = find_all_normals_naive(pts1, .7, flip_away=True)
    e2 = find_all_normals_naive(pts2, .7, flip_away=True)
    f = fit_KrigingSpline(pts1, pts1, e1, pts2, e2)
    f = fit_ThinPlateSpline(pts1, pts2)
if __name__ == "__main__":
    main()