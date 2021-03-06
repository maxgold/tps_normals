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
        dim = x.shape[0]
    assert x.shape[0] == dim and y.shape[0] == dim 
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
        return -r.T.dot(dr)/nr #-r.T.dot(dr)
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

def find_all_normals_naive (pcloud, wsize=0.02, flip_away=False, project_lower_dim=False):
    """
    Find normals at all the points.
    """
    if project_lower_dim:
        dim = pcloud.shape[1]
        pmean = pcloud.sum(axis=0)/pcloud.shape[0]
        p_centered = pcloud - pmean
        _,_,VT = np.linalg.svd(p_centered, full_matrices=True)
        p_lower_dim = VT[0:dim-1,:].dot(p_centered.T).T
        p_ld_nms = find_all_normals_naive(p_lower_dim,wsize=wsize,flip_away=flip_away,project_lower_dim=False)
        return VT[0:dim-1,:].T.dot(p_ld_nms.T).T

    
    normals = np.zeros([0,pcloud.shape[1]])
    for pt in pcloud:
        nm = find_normal_naive(pcloud,pt,wsize,flip_away)
        normals = np.r_[normals,np.atleast_2d(nm)]
    return normals