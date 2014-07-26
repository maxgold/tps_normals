from __future__ import division
import numpy as np, numpy.linalg as nlg
import scipy.spatial.distance as ssd

import cvxopt as co, cvxpy as cp

#from tn_utils import math_utils
from tn_rapprentice import tps
from tn_rapprentice.registration import ThinPlateSpline, fit_ThinPlateSpline, balance_matrix3

from tn_eval import tps_utils

from tn_eval.tps_evaluate import tps_eval

def loglinspace(a,b,n):
    "n numbers between a to b (inclusive) with constant ratio between consecutive numbers"
    return np.exp(np.linspace(np.log(a),np.log(b),n))


def tps_rpm_bij_normals_naive(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3,
                            nwsize = None, neps = None, plotting = False, plot_cb = None):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    Adding points for normals to fit tps to.
    Nothing fancy, just the baseline.

    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    """
    
    _,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)

    f = ThinPlateSpline(d)
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)
    
    g = ThinPlateSpline(d)
    g.trans_g = -f.trans_g

    # Calculating window sizes to find normals for points
    ndx = nlg.norm(x_nd.min(axis=0)-x_nd.max(axis=0))/x_nd.shape[0]
    ndy = nlg.norm(y_md.min(axis=0)-y_md.max(axis=0))/y_md.shape[0]
    nd = (ndx + ndy)/2
    if nwsize is None: nwsize = nd*2
    if neps is None: 
        neps = nd/2
    else:
        ndx = neps
        ndy = neps

    # Adds all normal points, at some small distance from points
    x_nms = x_nd + ndx/2*tps_utils.find_all_normals_naive(x_nd, ndx*2, flip_away= True, project_lower_dim=True)
    y_nms = y_md + ndy/2*tps_utils.find_all_normals_naive(y_md, ndy*2, flip_away= True, project_lower_dim=True)
    # The points to fit tps with
    xpts = np.r_[x_nd,x_nms]
    ypts = np.r_[y_md,y_nms]

    # r_N = None
    
    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        ywarped_md = g.transform_points(y_md)
        
        fwddist_nm = ssd.cdist(xwarped_nd, y_md,'euclidean')
        invdist_nm = ssd.cdist(x_nd, ywarped_md,'euclidean')
        
        r = rads[i]
        prob_nm = np.exp( -(fwddist_nm + invdist_nm) / (2*r) )
        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, 1e-1, 2e-1)
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)


        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)

        xt_nms = xtarg_nd + neps*tps_utils.find_all_normals_naive(xtarg_nd, nwsize, flip_away=True, project_lower_dim=True)
        yt_nms = ytarg_md + neps*tps_utils.find_all_normals_naive(ytarg_md, nwsize, flip_away=True, project_lower_dim=True)
        xtarg_pts = np.r_[xtarg_nd,xt_nms]
        ytarg_pts = np.r_[ytarg_md,yt_nms]
        
        wt_n_nm = np.r_[wt_n,wt_n]#/2
        wt_m_nm = np.r_[wt_m,wt_m]#/2

        
        if plotting and i%plotting==0 and plot_cb is not None:
            plot_cb(x_nd, y_md, xtarg_pts, corr_nm, wt_n, f)
        
        f = fit_ThinPlateSpline(xpts, xtarg_pts, bend_coef = regs[i], wt_n=wt_n_nm, rot_coef = rot_reg)
        g = fit_ThinPlateSpline(ypts, ytarg_pts, bend_coef = regs[i], wt_n=wt_m_nm, rot_coef = rot_reg)

    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_pts, regs[i], wt_n=wt_n_nm)/wt_n_nm.mean()
    g._cost = tps.tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ytarg_pts, regs[i], wt_n=wt_m_nm)/wt_m_nm.mean()
    return f,g


def calculate_normal_dist (x_na, y_ng, nwsize=0.04):
    e_x = tps_utils.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=True)
    e_y = tps_utils.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=True)
    return ssd.cdist(e_x,e_y,'euclidean')

def calculate_normal_dist2 (x_na, y_ng, nwsize=0.04):
    e_x = tps_utils.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=True)
    e_y = tps_utils.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=True)
#     import IPython
#     IPython.embed()
    return -e_x.dot(e_y.T)



def tps_rpm_bij_normals(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3, normal_coef=0.0001, 
                        nwsize=0.04, plotting = False, plot_cb = None, pts1 = None):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    """
    
    _,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)

    f = ThinPlateSpline(d)
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)
    
    g = ThinPlateSpline(d)
    g.trans_g = -f.trans_g


    # r_N = None
    
    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        ywarped_md = g.transform_points(y_md)
        
        fwddist_nm = ssd.cdist(xwarped_nd, y_md,'euclidean')
        fwddist_normals_nm = calculate_normal_dist2(xwarped_nd, y_md,nwsize)
        invdist_nm = ssd.cdist(x_nd, ywarped_md,'euclidean')
        invdist_normals_nm = calculate_normal_dist2(x_nd, ywarped_md, nwsize)
        
        #import IPython
        #IPython.embed()
        
        r = rads[i]
        prob_nm = np.exp( -(fwddist_nm + invdist_nm + normal_coef*(fwddist_normals_nm + invdist_normals_nm) / (2*r)))
        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, 1e-1, 2e-1)
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)


        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)
        
        if plotting and i%plotting==0 and plot_cb is not None:
            plot_cb(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f)
        
#         f = fit_ThinPlateSpline_normals(x_nd, xtarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg, normal_coef=normal_coef, nwsize = nwsize)
#         g = fit_ThinPlateSpline_normals(y_md, ytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg, normal_coef=normal_coef, nwsize = nwsize)
        f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)#, normal_coef=normal_coef, nwsize = nwsize)
        g = fit_ThinPlateSpline(y_md, ytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg)#, normal_coef=normal_coef, nwsize = nwsize)
#         print (f.transform_points(pts1))
        

    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, regs[i], wt_n=wt_n)/wt_n.mean()
    g._cost = tps.tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ytarg_md, regs[i], wt_n=wt_m)/wt_m.mean()
    return f,g


def fit_ThinPlateSpline_normals(x_na, y_ng, e_x = None, e_y = None, bend_coef=.1, rot_coef = 1e-5, normal_coef=1, wt_n=None,nwsize=0.02, use_cvx=False, use_dot=True):
    """
    x_na: source cloud
    y_nd: target cloud
    smoothing: penalize non-affine part
    angular_spring: penalize rotation
    wt_n: weight the points        
    """
    f = ThinPlateSpline()
    if use_cvx:
        f.lin_ag, f.trans_g, f.w_ng = tps_fit3_normals_cvx(x_na, y_ng, e_x, e_y, bend_coef, rot_coef, normal_coef, wt_n, use_dot=use_dot)
    else:
        f.lin_ag, f.trans_g, f.w_ng = tps_fit3_normals(x_na, y_ng, e_x, e_y, bend_coef, rot_coef, normal_coef, wt_n)
    f.x_na = x_na
    return f


def tps_fit3_normals(x_na, y_ng, e_x = None, e_y = None, bend_coef = .1, rot_coef = 1e-5, normal_coef = .01, wt_n = None, nwsize=0.02):
    if wt_n is None: wt_n = np.ones(len(x_na))
    n,d = x_na.shape

    
    K_nn = tps.tps_kernel_matrix(x_na)
    # Generate the normals
    if e_x is None:
        e_x = tps_utils.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=True)
    if e_y is None:
        e_y = tps_utils.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=True)
    
    # Calculate the relevant matrices for Jacobians
    if d == 3:
        x_diff = np.transpose(x_na[None,:,:] - x_na[:,None,:],(0,2,1))
        P = e_x.dot(x_diff)[range(n),range(n),:]/(K_nn+1e-20)
#         import IPython
#         IPython.embed()
    else:
        raise NotImplementedError
     

    Q = np.c_[np.ones((n,1)), x_na, K_nn]
    WQ = wt_n[:,None] * Q
    QWQ = Q.T.dot(WQ)
    H = QWQ
    H[d+1:,d+1:] += bend_coef * K_nn
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    H[1:d+1, 1:d+1] += np.diag(rot_coefs)

    ## Normals
    Qnr = np.c_[np.ones((n,1)), e_x, P]
    WQnr = wt_n[:,None] * Qnr
    QWQnr = Qnr.T.dot(WQnr)
    H += normal_coef*QWQnr 
    ##
    
    f = -WQ.T.dot(y_ng) - normal_coef*WQnr.T.dot(e_y)
    f[1:d+1,0:d] -= np.diag(rot_coefs)
    
    A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_na]].T
    
    Theta = tps.solve_eqp1(H,f,A)
    
    return Theta[1:d+1], Theta[0], Theta[d+1:]


def tps_fit3_normals_cvx(x_na, y_ng, e_x = None, e_y = None, bend_coef = .1, rot_coef = 1e-5, normal_coef = .01, wt_n = None, nwsize=0.02, use_dot=False):
    if wt_n is None: wt_n = np.ones(len(x_na))
    n,d = x_na.shape
    
    K_nn = tps.tps_kernel_matrix(x_na)
    _,_,VT = nlg.svd(np.c_[x_na,np.ones((x_na.shape[0],1))].T)
    Nmat = VT.T[:,d+1:]
    rot_coefs = np.diag(np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef)    
    
    # Generate the normals
    if e_x is None:
        e_x = tps_utils.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=True)
    if e_y is None:
        e_y = tps_utils.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=True)
    if d == 3:
        x_diff = np.transpose(x_na[None,:,:] - x_na[:,None,:],(0,2,1))
        Pmat = e_x.dot(x_diff)[range(n),range(n),:]/(K_nn+1e-20)
    else:
        raise NotImplementedError

    A = cp.Variable(Nmat.shape[1],d)
    B = cp.Variable(d,d)
    c = cp.Variable(d,1)
    
    X = co.matrix(x_na)
    Y = co.matrix(y_ng)
    EX = co.matrix(e_x)
    EY = co.matrix(e_y)
    
    K = co.matrix(K_nn)
    N = co.matrix(Nmat)
    P = co.matrix(Pmat)
    
    W = co.matrix(np.diag(wt_n))
    R = co.matrix(rot_coefs)
    ones = co.matrix(np.ones((n,1)))
    
    constraints = []
    
    # For correspondences
    V1 = cp.Variable(n,d)
    constraints.append(V1 == Y-K*N*A-X*B - ones*c.T)
    V2 = cp.Variable(n,d)
    constraints.append(V2 == cp.sqrt(W)*V1)
    # For normals
    if use_dot: 
#         import IPython
#         IPython.embed()
        N1 = cp.Variable(n,n)
        constraints.append(N1 == (P*N*A-EX*B)*EY.T)
        
#         N2 = cp.Variable(n)
#         constraints.extend([N2[i] == N1[i,i] for i in xrange(n)])
    else:
        N1 = cp.Variable(n,d)
        constraints.append(N1 == EY-P*N*A-EX*B)
        N2 = cp.Variable(n,d)
        constraints.append(N2 == cp.sqrt(W)*N1)
    # For bending cost
    Vb = []
    Q = [] # for quadratic forms
    for i in range(d):
        Vb.append(cp.Variable(Nmat.shape[1],1))
        constraints.append(Vb[-1] == A[:,i])
        Q.append(cp.quad_form(Vb[-1], N.T*K*N))
    # For rotation cost
    V3 = cp.Variable(d,d)
    constraints.append(V3 == cp.sqrt(R)*B)
    
    # Orthogonality constraints for bending
    constraints.extend([X.T*A == 0, ones.T*A == 0])
    
    # TPS objective
    if use_dot:
        objective = cp.Minimize(cp.sum_squares(V2) - normal_coef*sum([N1[i,i] for i in xrange(n)]) 
                                + bend_coef*sum(Q) + cp.sum_squares(V3))
    else:
        objective = cp.Minimize(cp.sum_squares(V2) + normal_coef*cp.sum_squares(N2) + bend_coef*sum(Q) + cp.sum_squares(V3))
     
    
    p = cp.Problem(objective, constraints)
    p.solve()
    
#     import IPython
#     IPython.embed()
    
    return np.array(B.value), np.squeeze(np.array(c.value)) , np.array(A.valuefi)

def tps_rpm_bij_normals_max(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3, normal_coef=0.0001, 
                        nwsize=.15, plotting = False, plot_cb = None):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    """
    
    _,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)

    f = ThinPlateSpline(d)
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)
    
    g = ThinPlateSpline(d)
    g.trans_g = -f.trans_g


    # r_N = None
    
    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        ywarped_md = g.transform_points(y_md)
        
        fwddist_nm = ssd.cdist(xwarped_nd, y_md,'euclidean')
        
        invdist_nm = ssd.cdist(x_nd, ywarped_md,'euclidean')
        
        
        
        
        r = rads[i]
        prob_nm = np.exp( -(fwddist_nm + invdist_nm  / (2*r)))
        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, 1e-1, 2e-1)
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)

        wt_n = np.array(wt_n, dtype='float')
        wt_m = np.array(wt_m, dtype='float')


        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)

        e_x = tps_utils.find_all_normals_naive(x_nd, nwsize, flip_away = True, project_lower_dim=(d==3))
        e_y = tps_utils.find_all_normals_naive(y_md, nwsize, flip_away = True, project_lower_dim=(d==3))
        e_xt = tps_utils.find_all_normals_naive(xtarg_nd, nwsize, flip_away = True, project_lower_dim=(d==3))
        e_yt = tps_utils.find_all_normals_naive(ytarg_md, nwsize, flip_away = True, project_lower_dim=(d==3))
        
        # if plotting and i%plotting==0 and plot_cb is not None:
        #    plot_cb(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f)
        
#         f = fit_ThinPlateSpline_normals(x_nd, xtarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg, normal_coef=normal_coef, nwsize = nwsize)
#         g = fit_ThinPlateSpline_normals(y_md, ytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg, normal_coef=normal_coef, nwsize = nwsize)
        f = tps_eval(x_nd, xtarg_nd, e_x, e_xt, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)#, normal_coef=normal_coef, nwsize = nwsize)
        g = tps_eval(y_md, ytarg_md, e_y, e_yt, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg)#, normal_coef=normal_coef, nwsize = nwsize)

#    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, regs[i], wt_n=wt_n)/wt_n.mean()
#f    g._cost = tps.tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ytarg_md, regs[i], wt_n=wt_m)/wt_m.mean()
    return f,g




def tps_rpm_bij_normals_max2(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3, normal_coef=0.0001, 
                        nwsize=.15, plotting = False, plot_cb = None):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    """
    
    _,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)

    f = ThinPlateSpline(d)
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)
    
    g = ThinPlateSpline(d)
    g.trans_g = -f.trans_g


    # r_N = None
    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        ywarped_md = g.transform_points(y_md)
        
        fwddist_nm = ssd.cdist(xwarped_nd, y_md,'euclidean')
        
        invdist_nm = ssd.cdist(x_nd, ywarped_md,'euclidean')
        
        r = rads[i]
        prob_nm = np.exp( -(fwddist_nm + invdist_nm  / (2*r)))
        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, 1e-1, 2e-1)
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)

        wt_n = np.array(wt_n, dtype='float')
        wt_m = np.array(wt_m, dtype='float')


        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)

        
        # if plotting and i%plotting==0 and plot_cb is not None:
        #    plot_cb(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f)
        
#         f = fit_ThinPlateSpline_normals(x_nd, xtarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg, normal_coef=normal_coef, nwsize = nwsize)
#         g = fit_ThinPlateSpline_normals(y_md, ytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg, normal_coef=normal_coef, nwsize = nwsize)
        f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)#, normal_coef=normal_coef, nwsize = nwsize)
        g = fit_ThinPlateSpline(y_md, ytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg)#, normal_coef=normal_coef, nwsize = nwsize)

    e_x = tps_utils.find_all_normals_naive(x_nd, nwsize, flip_away = True, project_lower_dim=(d==3))
    e_y = tps_utils.find_all_normals_naive(y_md, nwsize, flip_away = True, project_lower_dim=(d==3))
    e_xt = tps_utils.find_all_normals_naive(xtarg_nd, nwsize, flip_away = True, project_lower_dim=(d==3))
    e_yt = tps_utils.find_all_normals_naive(ytarg_md, nwsize, flip_away = True, project_lower_dim=(d==3))

    f = tps_eval(x_nd, xtarg_nd, e_x, e_xt, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)
    g = tps_eval(y_md, ytarg_md, e_y, e_yt, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg)

#    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, regs[i], wt_n=wt_n)/wt_n.mean()
#    g._cost = tps.tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ytarg_md, regs[i], wt_n=wt_m)/wt_m.mean()
    return f,g

