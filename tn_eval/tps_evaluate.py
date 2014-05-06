"""
Steps:

Working in 2D for now. Have to rederive this for 3D.

1. Compute the initial landmark-only spline to get a bunch of displacements at landmarks.
2. Find the slopes of edges given by this spline -- and find slope-difference displacements.
3. Find new spline based on the stuff in the paper -- get it back in old form.


Slope functions are old Kernels for landmarks + derivatives for slopes.
"""
import numpy as np, numpy.linalg as nlg
import scipy.linalg as slg
import cvxopt as co, cvxpy as cp


import tps_utils as tu
from tn_rapprentice import registration, tps

def tps_eval(x_na, y_ng, bend_coef, rot_coef, wt_n = None, nwsize=0.02, delta=0.02):
    """
    delta: Normal length.
    """
    n,dim = x_na.shape
    # Normals
    e_x = tu.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=True)
    e_y = tu.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=True)
    
    ## First, we solve the landmark only spline.
    f = registration.fit_ThinPlateSpline(x_na, y_ng, bend_coef, rot_coef, wt_n, use_cvx=True)
    
#     import IPython
#     IPython.embed()
    
    # What are the slope values caused by these splines at the points?
    # It can be found using the Jacobian at the point.
    # Finding what the normals are being mapped to
    d0 = np.empty((dim,0))
    for x, nm in zip(x_na,e_x):
        d0 = np.c_[d0,tu.tps_jacobian(f, x, dim).dot(nm)]
    d0 = d0.reshape((d0.shape[1]*dim,1))
    # Desired slopes
    d = e_y.T.reshape((e_y.shape[0]*dim,1))
    
    ## Let's find the difference of the slopes to get the edge correction.
    d_diff = d - d0
    
    # Finding the evaluation matrix.
    K = tu.tps_kernel_mat(x_na)
    Q = np.c_[np.ones((n,1)),x_na]
    # normal eval matrix
    L = np.r_[np.c_[K,Q],np.c_[Q.T,np.zeros((dim+1,dim+1))]]
    Linv = nlg.inv(L)
    
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
                M[i,j] = tu.deriv_U(p1,p2,n2,dim)
                if i < j:
                    P[i,j] = P[j,i] = tu.deriv2_U(p1,p2,n1,n2,dim)
    M = np.r_[M,np.zeros((1,n)),e_x.T]
    T  = np.r_[np.c_[np.eye(n+dim+1), np.zeros((n+dim+1,n))],np.c_[M.T.dot(Linv), np.eye(n)]]
    N = P + M.T.dot(Linv).dot(M) # + log(del/delta) ---> assuming all the normals are of same length
    
    # Evaluation matrix for just the change slopes
    Q = (-2*np.log(delta))*np.eye(dim*n) - slg.block_diag(*[N]*dim)
    
    # coefficients of orthogonalized slope elements
    w_diff = nlg.inv(Q).dot(d_diff)
    w_diff = w_diff.reshape((n,dim), order='F')
    # padding with 0's
    w_diff_whole = np.r_[np.zeros((n+dim+1,dim)),w_diff]
    w_whole = T.T.dot(w_diff_whole)
    
    w_final = np.r_[f.w_ng, np.atleast_2d(f.trans_g), f.lin_ag, np.zeros((n,dim))] + w_whole
    
    fn = registration.ThinPlateSplineNormals(dim)
    fn.x_na, fn.n_na = x_na, e_x
    fn.w_ng, fn.trans_g, fn.lin_ag, fn.wn_ng= w_final[:n,:], w_final[n,:], w_final[n+1:n+1+dim,:], w_final[n+1+dim:,:]
    
    
    import IPython
    IPython.embed()
    return fn 
    
def tps_fit3_normals_cvx(x_na, y_ng, bend_coef, rot_coef, normal_coef, wt_n, nwsize=0.02, use_dot=False):
    if wt_n is None: wt_n = np.ones(len(x_na))
    n,d = x_na.shape
    K_nn = tps.tps_kernel_matrix(x_na)
    rot_coefs = np.diag(np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef)    
    # Generate the normals
    e_x = tu.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=True)
    e_y = tu.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=True)
    if d == 3:
        x_diff = np.transpose(x_na[None,:,:] - x_na[:,None,:],(0,2,1))
        Pmat = e_x.dot(x_diff)[range(n),range(n),:]/(K_nn+1e-20)
    else:
        raise NotImplementedError

    A = cp.Variable(n,d)
    B = cp.Variable(d,d)
    c = cp.Variable(d,1)
    
    X = co.matrix(x_na)
    Y = co.matrix(y_ng)
    EX = co.matrix(e_x)
    EY = co.matrix(e_y)
    
    K = co.matrix(K_nn)
    K2 = co.matrix(np.sqrt(-K_nn))
    P = co.matrix(Pmat)
    
    W = co.matrix(np.diag(wt_n))
    R = co.matrix(rot_coefs)
    ones = co.matrix(np.ones((n,1)))
    
    constraints = []
    
    # For correspondences
    V1 = cp.Variable(n,d)
    constraints.append(V1 == Y-K*A-X*B - ones*c.T)
    V2 = cp.Variable(n,d)
    constraints.append(V2 == cp.sqrt(W)*V1)
    # For normals
    if use_dot: 
#         import IPython
#         IPython.embed()
        N1 = cp.Variable(n,n)
        constraints.append(N1 == (P*A-EX*B)*EY.T)
        
#         N2 = cp.Variable(n)
#         constraints.extend([N2[i] == N1[i,i] for i in xrange(n)])
    else:
        N1 = cp.Variable(n,d)
        constraints.append(N1 == EY-P*A-EX*B)
        N2 = cp.Variable(n,d)
        constraints.append(N2 == cp.sqrt(W)*N1)
    # For bending cost
    V3 = cp.Variable(n,d)
    constraints.append(V3 == K2*A)
    # For rotation cost
    V4 = cp.Variable(d,d)
    constraints.append(V4 == cp.sqrt(R)*B)
    
    # Orthogonality constraints for bending
    constraints.extend([X.T*A == 0, ones.T*A == 0])
    
    # TPS objective
    if use_dot:
        objective = cp.Minimize(sum(cp.square(V2)) - normal_coef*sum([N1[i,i] for i in xrange(n)]) 
                                + bend_coef*sum(cp.square(V3)) + sum(cp.square(V4)))
    else:
        objective = cp.Minimize(sum(cp.square(V2)) + normal_coef*sum(cp.square(N2)) + bend_coef*sum(cp.square(V3)) + sum(cp.square(V4)))
     
    
    p = cp.Problem(objective, constraints)
    p.solve()
    
#     import IPython
#     IPython.embed()
    
    return np.array(B.value), np.squeeze(np.array(c.value)) , np.array(A.value)