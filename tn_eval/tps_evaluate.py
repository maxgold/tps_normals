"""
Steps:

Working in 2D for now. Have to rederive this for 3D.

1. Compute the initial landmark-only spline to get a bunch of displacements at landmarks.
2. Find the slopes of edges given by this spline -- and find slope-difference displacements.
3. Find new spline based on the stuff in the paper -- get it back in old form.


Slope functions are old Kernels for landmarks + derivatives for slopes.
"""
from __future__ import division
import numpy as np, numpy.linalg as nlg
import scipy.linalg as slg
import cvxopt as co, cvxpy as cp


import tps_utils as tu
from tn_rapprentice import registration, tps

from cvxpy.atoms.affine.transpose import transpose


def transformed_normal_direction(x,ex,f,delta):
    y = f.tf -ransform_points(x)
    ey = (f.transform_points(x + delta*ex)-y)/delta
    return y, ey

def tps_eval(x_na, y_ng, e_x = None, e_y = None, bend_coef = 0.1, rot_coef = 1e-5, wt_n = None, nwsize=0.02, delta=0.0001):
    """
    delta: Normal length.
    """
    n,dim = x_na.shape
    
    # Finding the evaluation matrix.
    K = tu.tps_kernel_mat(x_na)
    Q1 = np.c_[np.ones((n,1)),x_na]
    # normal eval matrix1
    L = np.r_[np.c_[K,Q1],np.c_[Q1.T,np.zeros((dim+1,dim+1))]]
    Linv = nlg.inv(L)
    
    # Normals
    if e_x is None:
        e_x = tu.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=(dim==3))
    if e_y is None:
        e_y = tu.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=(dim==3))
    
    ## First, we solve the landmark only spline.
    f = registration.fit_ThinPlateSpline(x_na, y_ng, bend_coef=bend_coef, rot_coef=rot_coef, wt_n=wt_n, use_cvx=False)
    
    # What are the slope values caused by these splines at the points?
    # It can be found using the Jacobian at the point.
    # Finding what the normals are being mapped to
    d0 = np.empty((dim,0))
    for x, nm in zip(x_na,e_x):
        d0 = np.c_[d0,tu.tps_jacobian(f, x, dim).dot(nm)]
    
    
    # import IPython
    # IPython.embed()
    d0 = d0.reshape((d0.shape[1]*dim,1))

    # Desired slopes
    d = e_y.T.reshape((e_y.shape[0]*dim,1))
    

    
    ## Let's find the difference of the slopes to get the edge correction.
    d_diff = d - d0
    
    M = np.zeros((n,n))
    P = np.zeros((n,n))
    # Get rid of these for loops at some point
    for i in range(n):
        p1, n1 = x_na[i,:], e_x[i,:]
        for j in range(n):
            if i == j:
                M[i,i] = P[i,i] = 0
            else:
                p2, n2 = x_na[j,:], e_x[j,:]
                M[i,j] = tu.deriv_U(p2,p1,n2,dim)
                if i < j:
                    P[i,j] = P[j,i] = tu.deriv2_U(p1,p2,n2,n1,dim)
    M = np.r_[M,np.zeros((1,n)),e_x.T]
    T  = np.r_[np.c_[np.eye(n+dim+1), np.zeros((n+dim+1,n))],np.c_[-M.T.dot(Linv), np.eye(n)]]
    N = P + M.T.dot(Linv).dot(M) # + 2*log(del/delta) ---> assuming all the normals are of same length
    
    # Evaluation matrix for just the change slopes
    Q_single_dim = -2*np.log(delta)*(np.eye(n) +1.0/(2*np.log(delta))*N) # for single dimension
    Q = slg.block_diag(*[Q_single_dim]*dim)
    
    # coefficients of orthogonalized slope elements
    w_diff = nlg.inv(Q).dot(d_diff) # ----> This is where the shit happens
    w_diff = w_diff.reshape((n,dim), order='F')
    # padding with 0's
    w_diff_whole = np.r_[np.zeros((n+dim+1,dim)),w_diff]
    w_whole = T.T.dot(w_diff_whole)
    
    w_final = np.r_[f.w_ng, np.atleast_2d(f.trans_g), f.lin_ag, np.zeros((n,dim))] + w_whole
    
    fn = registration.ThinPlateSplineNormals(dim)
    fn.x_na, fn.n_na = x_na, e_x
    fn.w_ng, fn.trans_g, fn.lin_ag, fn.wn_ng= w_final[:n,:], w_final[n,:], w_final[n+1:n+1+dim,:], w_final[n+1+dim:,:]

    # import IPython
    # IPython.embed()
    return fn 


def tps_fit_normals_cvx(x_na, y_ng, e_x = None, e_y = None, bend_coef=0.1, rot_coef=1e-5, normal_coef = 0.1, wt_n=None, delta=0.0001, nwsize=0.02, point_coef = 1):
    """
    Fits normals and points all at once.
    delta: edge length
    """
    n,d = x_na.shape
    if wt_n is None: wt_n = co.matrix(np.ones(len(x_na)))

    # Normals
    if e_x is None:
        e_x = tu.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=(d==3))
    if e_y is None:
        e_y = tu.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=(d==3))

    K_nn = tu.tps_kernel_mat(x_na)
    Qmat = np.c_[np.ones((n,1)),x_na]
    Lmat = np.r_[np.c_[K_nn,Qmat],np.c_[Qmat.T,np.zeros((d+1,d+1))]]
    Mmat = np.zeros((n,n))
    Pmat = np.zeros((n,n))
    # Get rid of these for loops at some point
    for i in range(n):
        pi, ni = x_na[i,:], e_x[i,:]
        for j in range(n):
            if i == j:
                Mmat[i,i] = Pmat[i,i] = 0
            else:
                pj, nj = x_na[j,:], e_x[j,:]
                Mmat[i,j] = tu.deriv_U(pj,pi,nj,d)
                if i < j:
                    Pmat[i,j] = Pmat[j,i] = tu.deriv2_U(pi,pj,nj,ni,d)
    #Mmat = np.r_[Mmat,np.zeros((1,n)),e_x.T]
#     import IPython
#     IPython.embed()
    DKmat = -2*(np.diag([np.log(delta)]*n)) - Pmat
    Emat = np.r_[np.c_[K_nn, Mmat],np.c_[Mmat.T, DKmat]]
    
    # working with the kernel of the orthogonality constraints
    OCmat = np.r_[np.c_[x_na,np.ones((x_na.shape[0],1))], np.c_[e_x,np.zeros((e_x.shape[0],1))]].T
    _,_,VT = nlg.svd(OCmat)
    NSmat = VT.T[:,d+1:] #VT null space
    rot_coefs = np.diag(np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef)

    # if d == 3:
    #     x_diff = np.transpose(x_na[None,:,:] - x_na[:,None,:],(0,2,1))
    #     Pmat = e_x.dot(x_diff)[range(n),range(n),:]/(K_nn+1e-20)
    # else:
    #     raise NotImplementedError

    # A1 = cp.Variable(n,d) #f.w_ng
    # A2 = cp.Variable(n,d) #f.wn_ng
    A = cp.Variable(NSmat.shape[1],d) # stacked form of f.w_ng and f.wn_ng
    B = cp.Variable(d,d) #f.lin_ag
    c = cp.Variable(d,1) #f.trans_g
    
    X = co.matrix(x_na)
    Y = co.matrix(y_ng)
    EX = co.matrix(e_x)
    EY = co.matrix(e_y)

    NS = co.matrix(NSmat) # working in the null space of the constraints
    KM = co.matrix(np.c_[K_nn, Mmat])
    MDK = co.matrix(np.c_[Mmat.T,DKmat])
    E = co.matrix(Emat)
    
    wt_n
    W = co.matrix(np.diag(wt_n))
    R = co.matrix(rot_coefs)
    ones = co.matrix(np.ones((n,1)))

    constraints = []
    
    # For correspondences
    V1 = cp.Variable(n,d)
    constraints.append(V1 == KM*NS*A + X*B + ones*c.T - Y)
    V2 = cp.Variable(n,d)
    constraints.append(V2 == cp.sqrt(W)*V1)
    # For normals
    N1 = cp.Variable(n,d)
    constraints.append(N1 == MDK*NS*A+EX*B - EY)
    N2 = cp.Variable(n,d)
    constraints.append(N2 == cp.sqrt(W)*N1)
    # For bending cost
    Quad = [] # for quadratic forms
    for i in range(d):
        Quad.append(cp.quad_form(A[:,i], NS.T*E*NS))
    # For rotation cost
    V3 = cp.Variable(d,d)
    constraints.append(V3 == cp.sqrt(R)*B)

    
    


    
    # Orthogonality constraints for bending -- don't need these because working in the nullspace
    # constraints.extend([X.T*A1 +EX.T*A2== 0, ones.T*A1 == 0])
    
    # TPS objective
    objective = cp.Minimize(point_coef*cp.sum_squares(V2) + normal_coef*cp.sum_squares(N2) + bend_coef*sum(Quad) + cp.sum_squares(V3))
    #objective = cp.Minimize(cp.sum_squares(V2)  + bend_coef*sum(Quad) + cp.sum_squares(V3))

    
    p = cp.Problem(objective, constraints)
    p.solve(verbose=True)
    
    Aval = NSmat.dot(np.array(A.value))
    fn = registration.ThinPlateSplineNormals(d)
    fn.x_na, fn.n_na = x_na, e_x
    fn.w_ng, fn.wn_ng = Aval[0:n,:], Aval[n:,:]
    fn.trans_g, fn.lin_ag= np.squeeze(np.array(c.value)), np.array(B.value)
    #import IPython
    #IPython.embed()

    return fn


def tps_fit_normals_exact_cvx(x_na, y_ng, e_x = None, e_y = None, bend_coef=0.1, rot_coef=1e-5, normal_coef = 0.1, wt_n=None, delta=0.0001, nwsize=0.02):
    """
    Solves as basic a problem as possible from Bookstein --> no limits taken
    Fits normals and points all at once.
    delta: edge length
    """
    n,d = x_na.shape
    if wt_n is None: wt_n = co.matrix(np.ones(len(x_na)))

    # Normals
    if e_x is None:
        e_x = tu.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=(d==3))
    if e_y is None:
        e_y = tu.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=(d==3))
    
    xs_na = x_na# - e_x*delta/2
    xf_na = x_na + e_x*delta#/2

    Kmat = tps.tps_kernel_matrix(x_na)
    K1mat = tps.tps_kernel_matrix2(x_na, xs_na)
    K2mat = tps.tps_kernel_matrix2(x_na, xf_na)
    K12mat = tps.tps_kernel_matrix2(xs_na, xf_na)
    K11mat = tps.tps_kernel_matrix(xs_na)
    K22mat = tps.tps_kernel_matrix(xf_na)
    Qmat = np.c_[np.ones((n,1)),x_na]
    Q1mat = np.c_[np.ones((n,1)),xs_na]
    Q2mat = np.c_[np.ones((n,1)),xf_na]
    
    M1mat = np.r_[K1mat,Q1mat.T]
    M2mat = np.r_[K2mat,Q2mat.T]
    Dmat_inv = np.diag([1.0/delta]*n)
    
    MDmat = (M2mat - M1mat).dot(Dmat_inv)
    DKmat = Dmat_inv.dot(K11mat + K22mat - K12mat - K12mat.T).dot(Dmat_inv)
    
    Lmat = np.r_[np.c_[Kmat,Qmat],np.c_[Qmat.T,np.zeros((d+1,d+1))]]
    LEmat = np.r_[np.c_[Lmat, MDmat], np.c_[MDmat.T, DKmat]]


    # working with the kernel of the orthogonality constraints
    OCmat = np.r_[np.c_[x_na,np.ones((x_na.shape[0],1))], np.zeros((d+1,d+1)), np.c_[e_x,np.zeros((e_x.shape[0],1))]].T
    _,_,VT = nlg.svd(OCmat)
    NSmat = VT.T[:,d+1:] # null space
    rot_coefs = np.diag(np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef)


    # Problem setup:
    A = cp.Variable(NSmat.shape[1],d) #f.w_ng
    
    R = co.matrix(slg.block_diag(np.zeros((n+1,n+1)),rot_coefs, np.zeros((n,n))))
    NS = co.matrix(NSmat) # working in the null space of the constraints
    Y_EY = co.matrix(np.r_[y_ng,np.zeros((d+1,d)),e_y])
    LE = co.matrix(LEmat)
    
    constraints = []
    
    # For everything
    V1 = cp.Variable(2*n+d+1,d)
    constraints.append(V1 == LE*NS*A - Y_EY)
    # Bend cost
    Quad = [] # for quadratic forms
    for i in range(d):
        Quad.append(cp.quad_form(A[:,i], NS.T*LE*NS))
#     V = cp.Variable(d,d)
#     constraints.append(V == Y_EY.T*A)#Y.T*A1+EY.T*A2)
    V2 = cp.Variable(2*n+d+1,d)
    constraints.append(V2 == cp.sqrt(R)*NS*A)
    
    # TPS objective
    #objective = cp.Minimize(cp.sum_squares(V2) + normal_coef*cp.sum_squares(N2) + bend_coef*cp.sum_squares(V3) + cp.sum_squares(V4)
    objective = cp.Minimize(cp.sum_squares(LE*NS*A - Y_EY) + bend_coef*sum(Quad) + rot_coef*cp.sum_squares(cp.sqrt(R)*NS*A))

    p = cp.Problem(objective, constraints)
    p.solve(verbose=True)
     
    Aval = NSmat.dot(np.array(A.value)) 
    fn = registration.ThinPlateSplineNormals(d)
    fn.x_na, fn.n_na = x_na, e_x
    fn.w_ng, fn.trans_g, fn.lin_ag, fn.wn_ng= Aval[:n,:], Aval[n,:], Aval[n+1:n+1+d,:], Aval[n+1+d:,:]
    
    #import IPython
    #IPython.embed()
    

    return fn






def tps_fit_normals_cvx_test(x_na, y_ng, e_x = None, e_y = None, bend_coef=0.1, rot_coef=1e-5, normal_coef = 0.1, wt_n=None, delta=0.0001, nwsize=0.02, point_coef = 1):
    """
    Fits normals and points all at once.
    delta: edge length
    """
    n,d = x_na.shape
    if wt_n is None: wt_n = co.matrix(np.ones(len(x_na)))

    # Normals
    if e_x is None:
        e_x = tu.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=(d==3))
    if e_y is None:
        e_y = tu.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=(d==3))

    K_nn = tu.tps_kernel_mat(x_na)
    Qmat = np.c_[np.ones((n,1)),x_na]
    Lmat = np.r_[np.c_[K_nn,Qmat],np.c_[Qmat.T,np.zeros((d+1,d+1))]]
    Mmat = np.zeros((n,n))
    Pmat = np.zeros((n,n))
    # Get rid of these for loops at some point
    for i in range(n):
        pi, ni = x_na[i,:], e_x[i,:]
        for j in range(n):
            if i == j:
                Mmat[i,i] = Pmat[i,i] = 0
            else:
                pj, nj = x_na[j,:], e_x[j,:]
                Mmat[i,j] = tu.deriv_U(pj,pi,nj,d)
                if i < j:
                    Pmat[i,j] = Pmat[j,i] = tu.deriv2_U(pi,pj,nj,ni,d)
    #Mmat = np.r_[Mmat,np.zeros((1,n)),e_x.T]
#     import IPython
#     IPython.embed()
    DKmat = -2*(np.diag([np.log(delta)]*n)) - Pmat
    Emat = K_nn
    
    # working with the kernel of the orthogonality constraints
    OCmat = np.r_[np.c_[x_na,np.ones((x_na.shape[0],1))]].T
    _,_,VT = nlg.svd(OCmat)
    NSmat = VT.T[:,d+1:] #VT null space
    rot_coefs = np.diag(np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef)
    m = NSmat.shape[1]

    # if d == 3:
    #     x_diff = np.transpose(x_na[None,:,:] - x_na[:,None,:],(0,2,1))
    #     Pmat = e_x.dot(x_diff)[range(n),range(n),:]/(K_nn+1e-20)
    # else:
    #     raise NotImplementedError

    # A1 = cp.Variable(n,d) #f.w_ng
    # A2 = cp.Variable(n,d) #f.wn_ng
    A = cp.Variable(NSmat.shape[1] + n,d) # stacked form of f.w_ng f.wn_ng
    B = cp.Variable(d,d) #f.lin_ag
    c = cp.Variable(d,1) #f.trans_g
    
    X = co.matrix(x_na)
    Y = co.matrix(y_ng)
    EX = co.matrix(e_x)
    EY = co.matrix(e_y)

    NS = co.matrix(NSmat) # working in the null space of the constraints
    KM = co.matrix(K_nn)
    MDK = co.matrix(DKmat)
    E = co.matrix(Emat)
    DK = co.matrix(DKmat)
    M = co.matrix(Mmat)

    
    wt_n
    W = co.matrix(np.diag(wt_n))
    R = co.matrix(rot_coefs)
    ones = co.matrix(np.ones((n,1)))

    constraints = []
    
    # For correspondences
    V1 = cp.Variable(n,d)
    constraints.append(V1 == KM*NS*A[:m, :] + M*A[m:, :] + X*B + ones*c.T - Y)  # Why do we need c? is it implemented correctly? it only appears once in objective function
    V2 = cp.Variable(n,d)
    constraints.append(V2 == cp.sqrt(W)*V1)
    # For normals
    N1 = cp.Variable(n,d)
    constraints.append(N1 == DK*A[m:,:] - EY)
    N2 = cp.Variable(n,d)
    constraints.append(N2 == cp.sqrt(W)*N1)
    # For bending cost
    Quad = [] # for quadratic forms
    for i in range(d):
        Quad.append(cp.quad_form(A[:m,i], NS.T*E*NS))
    # For rotation cost
    V3 = cp.Variable(d,d)
    constraints.append(V3 == cp.sqrt(R)*B)

    
    


    
    # Orthogonality constraints for bending -- don't need these because working in the nullspace
    # constraints.extend([X.T*A1 +EX.T*A2== 0, ones.T*A1 == 0])
    
    # TPS objective
    objective = cp.Minimize(point_coef*cp.sum_squares(V2) + normal_coef*cp.sum_squares(N2) + bend_coef*sum(Quad) + cp.sum_squares(V3))
    #objective = cp.Minimize(cp.sum_squares(V2)  + bend_coef*sum(Quad) + cp.sum_squares(V3))

    
    p = cp.Problem(objective, constraints)
    p.solve(verbose=True)
    
    Aval = NSmat.dot(np.array(A[:m,:].value))
    fn = registration.ThinPlateSplineNormals(d)
    fn.x_na, fn.n_na = x_na, e_x
    fn.w_ng, fn.wn_ng = Aval, np.array(A.value[m:,:])
    fn.trans_g, fn.lin_ag= np.squeeze(np.array(c.value)), np.array(B.value)
    #import IPython
    #IPython.embed()

    return fn



def normal_exact(x_na, y_ng, e_x=None, e_y=None, delta=1e-4, nwsize=.02):

    n,d = x_na.shape

    # Normals
    if e_x is None:
        e_x = tu.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=(d==3))
    if e_y is None:
        e_y = tu.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=(d==3))

    Kmat = tps.tps_kernel_matrix2(x_na, x_na)
    KDmat = tps.tps_kernel_matrix2(x_na + delta*e_x, x_na)
    Dmat = tps.tps_normals_deriv_mat(x_na, x_na, e_x)
    DDmat = tps.tps_normals_deriv_mat(x_na + delta*e_x, x_na, e_x)

    K = co.matrix(Kmat/delta)
    KD = co.matrix(KDmat/delta)
    D = co.matrix(Dmat/delta)
    DD = co.matrix(DDmat/delta)
    EY = co.matrix(e_y)

    A = cp.Variable(2*n, d) #stacked f.w_ng, f.wn_ng

    constraints = []

    TD = cp.Variable(n,d)
    constraints.append(TD == KD*A[:n, :] + DD*A[n:, :])
    T = cp.Variable(n,d)
    constraints.append(T == K*A[:n,:] + D*A[n:, :])
    TDT = cp.Variable(n,d)
    constraints.append(TDT == TD - T) #transformed normal directions

    #TD = cp.Variable(n,d) # f.transform_points(x_na + delta*e_x)
    #constraints.append(TD == KD*NS*A[:m, :] + DD*A[m:, :] + XD*B)
    #T = cp.Variable(n,d) #f.transform_points(x_na)
    #constraints.append(T == Kd*NS*A[:m,:] + D*A[m:, :] + X*B)
    #TDT = cp.Variable(n,d) #transformed normal directions
    #constraints.append(TDT == TD - T) 

    # Objective is to minimize distance between transformed normal and desired normals
    objective = cp.Minimize(cp.sum_squares(TDT - EY))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver = cp.CVXOPT, verbose=True)

    Aval = np.array(A.value)
    fn = registration.ThinPlateSplineNormals(d)
    fn.x_na, fn.n_na = x_na, e_x
    fn.w_ng, fn.wn_ng = Aval[:n, :], Aval[n:, :]
    fn.trans_g, fn.lin_ag = np.zeros((1,d)), np.zeros((d,d))
    #fn.trans_g, fn.lin_ag = np.squeeze(np.array(c.value)), np.array(B.value)

    return fn



def new_tps_cvx(x_na, y_ng, e_x=None, e_y=None, point_coef=1, normal_coef=.1, nwsize=.02, wt_n=None, delta=1e-4, rot_coef=1e-5):

    # Right now fn.trans_g is missing

    n,d = x_na.shape
    if wt_n is None: wt_n = co.matrix(np.ones(len(x_na)))

    # Normals
    if e_x is None:
        e_x = tu.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=(d==3))
    if e_y is None:
        e_y = tu.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=(d==3))
 
    Mmat = tps.tps_normals_deriv_mat(x_na, x_na, e_x)
    Kmat = tps.tps_kernel_matrix2(x_na, x_na)
    KDmat = tps.tps_kernel_matrix2(x_na + delta*e_x, x_na)
    Dmat = tps.tps_normals_deriv_mat(x_na, x_na, e_x)
    DDmat = tps.tps_normals_deriv_mat(x_na + delta*e_x, x_na, e_x)

    # working with the kernel of the orthogonality constraints
    OCmat = np.r_[np.c_[x_na,np.ones((x_na.shape[0],1))]].T
    _,_,VT = nlg.svd(OCmat)
    NSmat = VT.T[:,d+1:] #VT null space
    rot_coefs = np.diag(np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef)
    m = NSmat.shape[1]
    
    Kd = co.matrix(Kmat/delta)
    KD = co.matrix(KDmat/delta)
    D = co.matrix(Dmat/delta)
    DD = co.matrix(DDmat/delta)
    EY = co.matrix(e_y)

    A = cp.Variable(NSmat.shape[1] + n,d) # stacked form of f.w_ng f.wn_ng
    B = cp.Variable(d,d) #f.lin_ag
    c = cp.Variable(d,1) #f.trans_g
    
    X = co.matrix(x_na)
    Xd = co.matrix(x_na/delta)
    XD = co.matrix((x_na + delta)/delta)
    Y = co.matrix(y_ng)
    EX = co.matrix(e_x)
    EY = co.matrix(e_y)

    NS = co.matrix(NSmat) # working in the null space of the constraints
    K = co.matrix(Kmat)
    M = co.matrix(Mmat)

    W = co.matrix(np.diag(wt_n))
    R = co.matrix(rot_coefs)
    ones = co.matrix(np.ones((n,1)))

    constraints = []


    # normals
    TD = cp.Variable(n,d) # f.transform_points(x_na + delta*e_x)
    constraints.append(TD == KD*NS*A[:m, :] + DD*A[m:, :])
    T = cp.Variable(n,d) #f.transform_points(x_na)
    constraints.append(T == Kd*NS*A[:m,:] + D*A[m:, :])
    TDT = cp.Variable(n,d) #transformed normal directions
    constraints.append(TDT == TD - T) 

    # correspondences 
    V1 = cp.Variable(n,d) 
    constraints.append(V1 == K*NS*A[:m, :] + M*A[m:, :] + X*B + ones*c.T - Y)  # Why do we need c? is it implemented correctly? it only appears once in objective function
    V2 = cp.Variable(n,d)
    constraints.append(V2 == cp.sqrt(W)*V1)

    # Objective is to minimize distance between transformed normal and desired normals
    objective = cp.Minimize(point_coef*cp.sum_squares(V2) + normal_coef*cp.sum_squares(TDT - EY))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver = cp.CVXOPT, verbose=True)

    Aval = NSmat.dot(np.array(A[:m, :].value))
    fn = registration.ThinPlateSplineNormals(d)
    fn.x_na, fn.n_na = x_na, e_x
    fn.w_ng, fn.wn_ng = Aval, np.array(A.value[m:, :])
    fn.trans_g, fn.lin_ag = np.squeeze(np.array(c.value)), np.array(B.value)

    # When you ignore the linear terms things work perfectly, but you have to ignore them in tps_eval_normals

    return fn




"""

def new_tps_cvx1(x_na, y_ng, e_x=None, e_y=None, point_coef=1, normal_coef=.1, nwsize=.02, wt_n=None, delta=1e-4, rot_coef=1e-5, bend_coef = .1):
    n,d = x_na.shape
    if wt_n is None: wt_n = co.matrix(np.ones(len(x_na)))

    # Normals
    if e_x is None:
        e_x = tu.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=(d==3))
    if e_y is None:
        e_y = tu.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=(d==3))
 
    Kmat = tps.tps_kernel_matrix2(x_na, x_na)
    KDmat = tps.tps_kernel_matrix2(x_na + delta*e_x, x_na)
    Dmat = tps.tps_normals_deriv_mat(x_na, x_na, e_x)
    DDmat = tps.tps_normals_deriv_mat(x_na + delta*e_x, x_na, e_x)

    Mmat = np.zeros((n,n))
    Pmat = np.zeros((n,n))
    for i in range(n):
        pi, ni = x_na[i,:], e_x[i,:]
        for j in range(n):
            if i == j:
                Mmat[i,i] = Pmat[i,i] = 0
            else:
                pj, nj = x_na[j,:], e_x[j,:]
                Mmat[i,j] = tu.deriv_U(pj,pi,nj,d)
                if i < j:
                    Pmat[i,j] = Pmat[j,i] = tu.deriv2_U(pi,pj,nj,ni,d)
    DKmat = -2*(np.diag([np.log(delta)]*n)) - Pmat
    Emat = np.r_[np.c_[Kmat, Mmat],np.c_[Mmat.T, DKmat]]

    # working with the kernel of the orthogonality constraints
    #OCmat = np.r_[np.c_[x_na,np.ones((x_na.shape[0],1))]].T
    #_,_,VT = nlg.svd(OCmat)
    #NSmat = VT.T[:,d+1:] #VT null space
    #m = NSmat.shape[1]
    
    rot_coefs = np.diag(np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef)
    OCmat = np.r_[np.c_[x_na,np.ones((x_na.shape[0],1))], np.c_[e_x,np.zeros((e_x.shape[0],1))]].T
    _,_,VT = nlg.svd(OCmat)
    NSmat = VT.T[:,d+1:]

    Kd = co.matrix(Kmat/delta)
    KD = co.matrix(KDmat/delta)
    D = co.matrix(Dmat/delta)
    DD = co.matrix(DDmat/delta)
    KDD = co.matrix(np.c_[KDmat, DDmat]/delta)
    KdD = co.matrix(np.c_[Kmat, Dmat]/delta)
    EY = co.matrix(e_y)
    E = co.matrix(Emat)

    A = cp.Variable(NSmat.shape[1],d) # stacked form of f.w_ng f.wn_ng
    B = cp.Variable(d,d) #f.lin_ag
    c = cp.Variable(d,1) #f.trans_g
    
    X = co.matrix(x_na)
    Xd = co.matrix(x_na/delta)
    XD = co.matrix((x_na + delta)/delta)
    Y = co.matrix(y_ng)
    EX = co.matrix(e_x)
    EY = co.matrix(e_y)

    NS = co.matrix(NSmat) # working in the null space of the constraints
    K = co.matrix(Kmat)
    M = co.matrix(Mmat)
    KM = co.matrix(np.c_[Kmat, Mmat])

    W = co.matrix(np.diag(wt_n))
    R = co.matrix(rot_coefs)
    ones = co.matrix(np.ones((n,1)))

    constraints = []


    # normals
    TD = cp.Variable(n,d) # f.transform_points(x_na + delta*e_x)
    constraints.append(TD == KDD*NS*A + XD*B)
    T = cp.Variable(n,d) # f.transform_points(x_na)
    constraints.append(T == KdD*NS*A + Xd*B)
    TDT = cp.Variable(n,d) # transformed normal directions
    constraints.append(TDT == TD - T) 

    # correspondences 
    V1 = cp.Variable(n,d) 
    constraints.append(V1 == KM*NS*A + X*B + ones*c.T - Y)  # Why do we need c? is it implemented correctly? it only appears once in objective function
    V2 = cp.Variable(n,d)
    constraints.append(V2 == cp.sqrt(W)*V1)
    # For bending cost
    # Quad = [] # for quadratic forms
    # for i in range(d):
    #    Quad.append(cp.quad_form(A[:,i], NS.T*E*NS))
    # For rotation cost
    V3 = cp.Variable(d,d)
    constraints.append(V3 == cp.sqrt(R)*B)

    # Objective is to minimize distance between transformed normal and desired normals
    objective = cp.Minimize(point_coef*cp.sum_squares(V2) + normal_coef*cp.sum_squares(TDT - EY) ) + cp.sum_squares(V3)) # + bend_coef*sum(Quad))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver = cp.CVXOPT, verbose=True)

    Aval = NSmat.dot(np.array(A.value))
    fn = registration.ThinPlateSplineNormals(d)
    fn.x_na, fn.n_na = x_na, e_x
    fn.w_ng, fn.wn_ng = Aval[:n, :], Aval[n:, :]
    fn.trans_g, fn.lin_ag = np.squeeze(np.array(c.value)), np.array(B.value)

    # When you ignore the linear terms things work perfectly, but you have to ignore them in tps_eval_normals

    return fn
"""



def tps_fit3_cvx_normals(x_na, y_ng, e_x=None, e_y=None, bend_coef=.1, rot_coef=1e-5, wt_n=None, normal_coef=1, nwsize=.15, delta=.15):
    """
    Use cvx instead of just matrix multiply.
    Working with null space of matrices.
    """

    if e_x is None:
        e_x = tu.find_all_normals_naive(x_na, nwsize, flip_away=True)
    if e_y is None:
        e_y = tu.find_all_normals_naive(y_ng, nwsize, flip_away=True)

    if wt_n is None: wt_n = co.matrix(np.ones(len(x_na)))
    n,d = x_na.shape
    K_nn = tps.tps_kernel_matrix(x_na)
    x_na[.5*n:, :] *= normal_coef**.5
    y_ng[.5*n:, :] *= normal_coef**.5
    _,_,VT = nlg.svd(np.c_[x_na,np.ones((x_na.shape[0],1))].T)
    Nmat = VT.T[:,d+1:]
    rot_coefs = np.diag(np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef)
    
    
    A = cp.Variable(Nmat.shape[1],d)
    B = cp.Variable(d,d)
    c = cp.Variable(d,1)
    
    Y = co.matrix(y_ng)
    K = co.matrix(K_nn)
    N = co.matrix(Nmat)
    X = co.matrix(x_na)
    W = co.matrix(np.diag(wt_n).copy())
    R = co.matrix(rot_coefs)
    ones = co.matrix(np.ones((n,1)))
    
    constraints = []
    
    # For correspondences
    V1 = cp.Variable(n,d)
    constraints.append(V1 == Y-K*N*A-X*B - ones*c.T)
    V2 = cp.Variable(n,d)
    constraints.append(V2 == cp.sqrt(W)*V1)
    # For bending cost
    Q = [] # for quadratic forms
    for i in range(d):
        Q.append(cp.quad_form(A[:,i], N.T*K*N))
    # For rotation cost
    # Element wise square root actually works here as R is diagonal and positive
    V3 = cp.Variable(d,d)
    constraints.append(V3 == cp.sqrt(R)*B)
    
    # Orthogonality constraints for bending are taken care of already because working with the null space
    #constraints.extend([X.T*A == 0, ones.T*A == 0])
    
    # TPS objective
    objective = cp.Minimize(cp.sum_squares(V2) + bend_coef*sum(Q) + cp.sum_squares(V3))
    p = cp.Problem(objective, constraints)
    p.solve(verbose=True)

    Aval = Nmat.dot(np.array(A.value))
    fn = registration.ThinPlateSpline(d)
    fn.x_na = x_na
    fn.w_ng = Aval
    fn.trans_g, fn.lin_ag= np.squeeze(np.array(c.value)), np.array(B.value)


    
    return fn


def tps_fit3_normals(x_na, y_ng, e_x=None, e_y=None, bend_coef=.1, rot_coef=1e-5, wt_n = None, normal_coef=1, nwsize=.15, delta = .15):
    
    if e_x is None:
        e_x = tu.find_all_normals_naive(x_na, nwsize, flip_away=True)
    if e_y is None:
        e_y = tu.find_all_normals_naive(y_ng, nwsize, flip_away=True)
    x_na = np.r_[x_na, (normal_coef**.5)*(x_na + delta*e_x)]
    y_ng = np.r_[y_ng, (normal_coef**.5)*(y_ng + delta*e_y)]

    if wt_n is None: wt_n = np.ones(len(x_na))
    n,d = x_na.shape

    K_nn = tps.tps_kernel_matrix(x_na)
    Q = np.c_[np.ones((n,1)), x_na, K_nn]
    WQ = wt_n[:,None] * Q
    QWQ = Q.T.dot(WQ)
    H = QWQ
    H[d+1:,d+1:] += bend_coef * K_nn
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    H[1:d+1, 1:d+1] += np.diag(rot_coefs)
    
    f = -WQ.T.dot(y_ng)
    f[1:d+1,0:d] -= np.diag(rot_coefs)
    
    A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_na]].T
    
    Theta = tps.solve_eqp1(H,f,A)
    
    fn = registration.ThinPlateSpline(d)
    fn.x_na = x_na
    fn.lin_ag, fn.trans_g, fn.w_ng = Theta[1:d+1], Theta[0], Theta[d+1:]

    return fn















