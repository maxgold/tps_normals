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

import tps_utils as tu
from tn_rapprentice import registration

def tps_eval(x_na, y_ng, bend_coeff, rot_coeff, wt_n = None, nwsize=0.02, delta=0.02):
    """
    delta: Normal length.
    """
    n,dim = x_na.shape
    # Normals
    e_x = tu.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=True)
    e_y = tu.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=True)
    
    ## First, we solve the landmark only spline.
    f = registration.fit_ThinPlateSpline(x_na, y_ng, bend_coeff, rot_coeff, wt_n, use_cvx=True)
    
    import IPython
    IPython.embed()
    
    # What are the slope values caused by these splines at the points?
    # It can be found using the Jacobian at the point.
    # Finding what the normals are being mapped to
    d0 = np.empty((dim,0))
    for x,n in zip(x_na,e_x):
        d0 = np.c_[d0,tu.tps_jacobian(f, x, dim).dot(n)]
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
        for j in range(n):
            if i == j:
                M[i,i] = P[i,i] = 0
            else:
                p1, p2, n1, n2 = x_na[i,:],x_na[j,:],e_x[i,:],e_x[j,:]
                M[i,j] = tu.deriv_U(p1,p2,n2,dim)
                if i < j:
                    P[i,j] = P[j,i] = tu.deriv2_U(p1,p2,n1,n2,dim)
    M = np.r_[M,np.zeros((1,n)),e_x.T]
    T  = np.r_[np.c_[np.eye(n+3), np.zeros((n+3,n))],np.c_[M.T.dot(Linv), np.eye(n)]]
    N = P + M.T.dot(Linv).dot(M) # + log(del/delta) ---> assuming all the normals are of same length
    
    # Evaluation matrix for just the change slopes
    Q = (-2*np.log(delta))*np.eye(dim*n) - slg.block_diag(*[N]*dim)
    
    # coefficients of orthogonalized slope elements
    w_diff = nlg.inv(Q).dot(d_diff)
    w_diff = w_diff.reshape((n,dim), order='F')
    # padding with 0's
    w_diff_whole = np.r_[np.zeros((n+3,dim)),w_diff]
    w_whole = T.T.dot(w_diff_whole)
    
    w_final = 
    
    