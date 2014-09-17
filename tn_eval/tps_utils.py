###
# Adapted from: A Thin Plate Spline for deformations with specified derivatives. 
###
import numpy as np, numpy.linalg as nlg
import scipy.spatial.distance as ssd

def tps_kernel(x,y,dim=None):
    """
    Kernel function for tps.
    Only dim = 2 and 3 implemented.
    """
    if dim is None:
        dim = x.shape[1]
    assert x.shape[1] == dim and y.shape[1] == dim 
    if dim==2:
        r = nlg.norm(x-y)
        return r**2 * np.log(r+1e-20)
    elif dim==3:
        return -nlg.norm(x-y)
    else:
        raise NotImplementedError


def tps_kernel_mat (Xs):
    """
    Returns the kernel matrix of a list of points.
    """
    dim = Xs.shape[1]
    dist_mat = ssd.squareform(ssd.pdist(Xs))
    if dim == 3:
        return -dist_mat
    elif dim == 2:
        return dist_mat**2 * np.log(dist_mat+1e-20)
    else:
        raise NotImplementedError
    

def deriv_U(x,y,dr,dim=None):
    """
    First derivative for kernel: U'(x-y) in the direction of dr
    Only dim = 2 and 3 implemented.
    """
    if dim is None:
        dim = x.shape[0]
    assert x.shape[0] == dim and y.shape[0] == dim and dr.shape[0] == dim
    r = (x-y)
    nr = nlg.norm(r) 

    if dim==2:
        if nr == 0: return 0
        return (2*np.log(nr)+1)*(r.T.dot(dr))
    elif dim==3:
        if nr == 0: return 0
        return -r.T.dot(dr) #/nr -r.T.dot(dr)
    else:
        raise NotImplementedError
    

def deriv2_U(x,y,dr1,dr2,dim=None):
    """
    Second derivative for kernel.
    Only dim = 2 and 3 implemented.
    """
    if dim is None:
        dim = x.shape[0]
    assert x.shape[0] == dim and y.shape[0] == dim and dr1.shape[0] == dim and dr2.shape[0] == dim
    r = (x-y)
    nr = nlg.norm(r)
 
    if dim==2:
        if nr == 0: return 0
        return 2.0/(nr**2)*(r.T.dot(dr1))*(r.T.dot(dr2)) + (2*np.log(nr)+1)*(dr1.T.dot(dr2))  
    elif dim==3:
        if nr == 0: return 0
        return (r.T.dot(dr1))*(r.T.dot(dr2))/(nr**3) - (dr1.T.dot(dr2))/nr#-dr1.T.dot(dr2)
    else:
        raise NotImplementedError

def tps_jacobian_single_term (pt, jpt, dim=None):
    """
    Finds Jacobian of single term
    """
    if dim is None:
        dim = len(pt)
    assert len(pt) == dim and len(jpt) == dim
    
    r = (pt - jpt)
    nr = nlg.norm(r)
    
    if dim == 3:
        if nr == 0: return np.ones_like(r)
        return -r/nr
    elif dim == 2:
        if nr == 0: return np.zeros_like(r)
        return r*(2*np.log(nr) + 1)
    else: raise NotImplementedError
        
    
def tps_jacobian (f, pt, dim=None):
    """
    Finds the Jacobian Matrix at the point pt.
    """
    if dim is None:
        dim = len(pt)
    assert len(pt) == dim
    jac = np.copy(f.lin_ag.T)    
    for jpt, w in zip(f.x_na, f.w_ng):
        jac += np.atleast_2d(w).T.dot(np.atleast_2d(tps_jacobian_single_term(pt, jpt, dim)))

    return jac
    
    
def project_lower_dim (pcloud): 
    """
    Projects points into lower dimension (dim - 1)
    """
    dim = pcloud.shape[1]
    pmean = pcloud.sum(axis=0)/pcloud.shape[0]
    p_centered = pcloud - pmean
    _,_,VT = np.linalg.svd(p_centered, full_matrices=True)
    return VT[0:dim-1,:].dot(p_centered.T).T


def find_normal_naive (pcloud, pt, wsize=0.02,flip_away=False):
    """
    Selects close points on pcloud within windowsize and then does PCA to find normals.
    Normal could potentially be flipped.
    """
    dim = pcloud.shape[1]
    cpoints = pcloud[nlg.norm(pcloud-pt, axis=1) <= wsize,:]
    if cpoints.shape[0] < dim: return np.zeros(dim  )
    
    cpoints = cpoints - cpoints.sum(axis=0)/cpoints.shape[0]
    _,_,v = np.linalg.svd(cpoints, full_matrices=True)
    nm = v[dim-1,:].T

    #Potentially flip the normal: if more than half of the other points are in the direction of the normal, flip
    if flip_away:
        if sum((pcloud-pt).dot(nm)>0) > pcloud.shape[0]*1.0/2.0: nm = -nm
    
    return nm

def find_all_normals_naive (dspcloud, orig_cloud = None, wsize=0.02, flip_away=False, project_lower_dim=False):
    """
    Find normals at all the points of the downsampled point cloud, dspcloud, in the original point cloud, pcloud.
    """
    if orig_cloud is None:
        orig_cloud = dspcloud

    if project_lower_dim:
        dim = dspcloud.shape[1]
        pmean = dspcloud.sum(axis=0)/dspcloud.shape[0]
        p_centered = dspcloud - pmean
        _,_,VT = np.linalg.svd(p_centered, full_matrices=True)
        p_lower_dim = VT[0:dim-1,:].dot(p_centered.T).T
        p_ld_nms = find_all_normals_naive(p_lower_dim, orig_cloud = orig_cloud, wsize=wsize,flip_away=flip_away,project_lower_dim=False)
        return VT[0:dim-1,:].T.dot(p_ld_nms.T).T

    
    normals = np.zeros([0,dspcloud.shape[1]])
    for pt in dspcloud:
        nm = find_normal_naive(orig_cloud,pt,wsize,flip_away)
        normals = np.r_[normals,np.atleast_2d(nm)]
    return normals

def find_curvature_pt(pcloud, pt, wsize = 0.02):
    dim = pcloud.shape[1]
    cpoints = pcloud[nlg.norm(pcloud-pt, axis=1) <= wsize,:]

    if cpoints.shape[0] < dim: return 0
    
    cpoints = cpoints - cpoints.sum(axis=0)/cpoints.shape[0]
    _,S,_ = np.linalg.svd(cpoints, full_matrices=True)

    if np.sum(S) > 0:
        return S[-1]/np.sum(S)
    else:
        return 0

        
def find_all_curvatures(dspcloud, orig_cloud = None, wsize = 0.02, project_lower_dim = False):
    if orig_cloud is None:
        orig_cloud = dspcloud
    
    if project_lower_dim:
        dim = dspcloud.shape[1]
        pmean = dspcloud.sum(axis=0)/dspcloud.shape[0]
        p_centered = dspcloud - pmean
        _,_,VT = np.linalg.svd(p_centered, full_matrices=True)
        p_lower_dim = VT[0:dim-1,:].dot(p_centered.T).T
        p_ld_nms = find_all_curvatures(p_lower_dim, orig_cloud = orig_cloud, wsize=wsize, project_lower_dim=False)
        return VT[0:dim-1,:].T.dot(p_ld_nms.T).T

    curvatures = np.zeros([0])
    for pt in dspcloud:
        curve = find_curvature_pt(orig_cloud, pt, wsize)
        curvatures = np.r_[curvatures, curve]

    return curvatures

def find_all_normals_below_threshold(dspcloud, orig_cloud = None, thresh = .36, wsize = .1):
    if orig_cloud is None:
        orig_cloud = dspcloud
    _, d = dspcloud.shape

    normals = np.zeros([0,dspcloud.shape[1]])    
    if d == 3:
        znormal = np.array([0,0,1])
    else:
        znormal = np.array([0,0])
    for pt in dspcloud:
        curve = find_curvature_pt(orig_cloud, pt, wsize)
        if curve < thresh:
            nm = find_normal_naive(orig_cloud, pt, wsize)
            normals = np.r_[normals, np.atleast_2d(nm)]
        else:
            normals = np.r_[normals, np.atleast_2d(znormal)]

    return normals

def flip_normals(pts1, epts, exs,  pts2, eys, bend_coef = .1, pasta = None):
    from tn_rapprentice.registration import fit_KrigingSpline
    f = fit_KrigingSpline(pts1, epts, exs, pts2, eys, bend_coef = .1, normal_coef = 0)
    ewarped = f.transform_normals(epts, exs)
    eflipped = -eys
    ediff_n = nlg.norm(ewarped - eys, axis = 1)
    ediff_flip = nlg.norm(ewarped - eflipped, axis = 1)

    ediff = np.minimum(ediff_n, ediff_flip)


    bools1 = np.array([ediff_n - ediff], dtype = bool) + 1
    bools2 = np.array([ediff_flip - ediff], dtype = bool) + 1

    bools1 = bools1 %2 
    bools2 = bools2 %2

    bools1, bools2 = np.array(bools1, dtype = bool), np.array(bools2, dtype = bool)

    #import IPython as ipy
    #ipy.embed()

    final_flipped = eys*bools1.T + eflipped*bools2.T
    if pasta is not None:
        return final_flipped, f
    else:
        return final_flipped

def angle_difference(x, y):
    xy = np.dot(x, y)
    lx = nlg.norm(x)
    ly = nlg.norm(y)
    r = np.arccos(xy/(lx*ly))
    return r*180/np.pi

def ith_row(mat):
    n, _, d = mat.shape
    result_mat = np.zeros([0,d])
    for i in range(len(mat)):
        result_mat = np.r_[result_mat, mat[i][i][None, :]]
    return result_mat


def normal_corr_mult(corr, eys):
    #eys have to be normalized
    eys_n = eys.copy()
    eys_n /= nlg.norm(eys, axis=1)[:,None]
    n, d = eys.shape
    prelim = np.max(corr, axis=1)
    corr_bools = corr >= np.tile(prelim[:,None], ((1, n)))
    for i in range(len(corr_bools)):
        a = np.where(corr_bools[i])
        if len(a[0]) > 0:
            corr_bools[i][(a[0][0]+1):] *= 0
    corr_bools = np.c_[corr_bools, corr_bools].reshape((n*d, n))

    normals = np.tile(eys_n.T, ((n, 1)))
    #only care about first True value
    reference_normals = normals[corr_bools].reshape((n, d))
    eys_n_tiled = np.tile(eys_n, ((n,1))).reshape(n, n, d)
    reference_normals_tiled = np.tile(reference_normals, ((1,n))).reshape(n,n,d)

    check = np.sum((reference_normals_tiled*eys_n_tiled).reshape((n*n,d)), axis=1)
    check_bool = check >= 0
    check_bool = np.tile(check_bool[:,None], ((1,2))).reshape((n,n,d))
    check_bool_flip = check < 0
    check_bool_flip = np.tile(check_bool_flip[:,None], ((1,2))).reshape((n,n,d))
    eys_tiled = np.tile(eys, ((n,1))).reshape(n, n, d)
    final_eys = eys_tiled*check_bool - eys_tiled*check_bool_flip

    weighted_sum = np.zeros([0,d])
    for i in range(len(corr)):
        weighted_sum = np.r_[weighted_sum, corr[i].dot(final_eys[i])[None,:]]
    #import IPython
    #IPython.embed()

    return weighted_sum

def compute_curvature_weights(x_nd, y_md, wsize = .1):
    x_curves = find_all_curvatures(x_nd, wsize = wsize)
    y_curves = find_all_curvatures(y_md, wsize = wsize)
    sx_curves = .5 - x_curves
    sy_curves = .5 - y_curves
    curves_mat = 16*np.square(sx_curves[:,None]*sy_curves[None,:])

    return curves_mat











    
