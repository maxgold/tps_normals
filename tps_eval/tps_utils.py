###
# Adapted from: A Thin Plate Spline for deformations with specified derivatives. 
###
import numpy as np, numpy.linalg as nlg
import scipy.spatial.distance as ssd

def tps_kernel(x,y,dim=3):
    """
    Kernel function for tps.
    Only dim = 2 and 3 implemented.
    """
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
    

def deriv_U(x,y,dr,dim=3):
    """
    First derivative for kernel.
    Only dim = 2 and 3 implemented.
    """
    assert x.shape[0] == dim and y.shape[0] == dim and dr.shape[0] == dim
    r = (x-y)
    nr = nlg.norm(r) 

    if nr == 0: return 0
    if dim==2:
        return (2*np.log(nr)+1)*(r.T.dot(dr))
    elif dim==3:
        return -r.T.dot(dr)
    else:
        raise NotImplementedError
    

def deriv2_U(x,y,dr1,dr2,dim=3):
    """
    Second derivative for kernel.
    Only dim = 2 and 3 implemented.
    """
    assert x.shape[0] == dim and y.shape[0] == dim and dr1.shape[0] == dim and dr2.shape == dim
    r = (x-y)
    nr = nlg.norm(r)
 
    if nr == 0: return 0
    if dim==2:
        return 2.0/(nr**2)*(r.T.dot(dr1))*(r.T.dot(dr2)) + (2*np.log(nr)+1)*(dr1.T.dot(dr2))  
    elif dim==3:
        return -dr1.T.dot(dr2)
    else:
        raise NotImplementedError


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

def find_all_normals_naive (pcloud, wsize=0.02, flip_away=False):
    """
    Find normals at all the points.
    """
    normals = np.zeros([0,pcloud.shape[1]])
    for pt in pcloud:
        nm = find_normal_naive(pcloud,pt,wsize,flip_away)
        normals = np.r_[normals,np.atleast_2d(nm)]
    return normals