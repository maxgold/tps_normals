###
# Adapted from: A Thin Plate Spline for deformations with specified derivatives. 
###
import numpy as np, numpy.linalg as nlg

def tps_kernel(x,y,dim=3):
    """
    Kernel function for tps.
    """
    assert x.shape[0] == dim and y.shape[0] == dim 
    if dim==2:
        r = nlg.norm(x-y)
        return r**2 * np.log(r+1e-20)
    elif dim==3:
        return -nlg.norm(x-y)
    else:
        raise NotImplementedError


def deriv_U(x,y,dr,dim=3):
    """
    First derivative for kernel.
    """
    assert x.shape[0] == dim and y.shape[0] == dim and dr.shape[0] == dim
    r = (x-y)
    if dim==2:
        return (2*np.log(nlg.norm(r))+1)*(r.T.dot(dr))
    elif dim==3:
        return -r.T.dot(dr)
    else:
        raise NotImplementedError
    

def deriv2_U(x,y,dr,dim=3):
    """
    First derivative for kernel.
    """
    assert x.shape[0] == dim and y.shape[0] == dim and dr.shape[0] == dim
    r = (x-y)
    if dim==2:
        return (2*np.log(nlg.norm(r))+1)*(r.T.dot(dr))
    elif dim==3:
        return -r.T.dot(dr)
    else:
        raise NotImplementedError