import numpy as np, numpy.linalg as nlg
import scipy as sp, scipy.spatial.distance as ssd
from math import floor

from tn_eval.tps_utils import find_all_normals_naive

def nan2zero(x):
    np.putmask(x, np.isnan(x), 0)
    return x

def solve_eqp1(H, f, A):
    """solve equality-constrained qp
    min tr(x'Hx) + sum(f'x)
    s.t. Ax = 0
    """    
    n_vars = H.shape[0]
    assert H.shape[1] == n_vars
    assert f.shape[0] == n_vars
    assert A.shape[1] == n_vars
    n_cnts = A.shape[0]
    
    _u,_s,_vh = np.linalg.svd(A.T)
    N = _u[:,n_cnts:]
    # columns of N span the null space
    
    # x = Nz
    # then problem becomes unconstrained minimization .5*z'NHNz + z'Nf
    # NHNz + Nf = 0
    z = np.linalg.solve(N.T.dot(H.dot(N)), -N.T.dot(f))
    x = N.dot(z)
    
    return x

def sigma (alpha, x, y = None):
	"""
	Basis function for kriging.
	Alpha is paramater. 

	input is a n x d array

	Alpha is not an integer
	"""

	if y is None:
		y = np.zeros(x.shape)
	assert x.shape == y.shape
	l = x - y
	nl = np.sum(np.abs(l)**2,axis=-1)**(1./2)
	return (-1)**(floor(alpha)+1)*nl**(2*alpha)


def krig_mat_landmark (alpha, Xs):
	"""
	Returns the krig-basis matrix of a list of points
	when there are only landmark constraints
	"""
	dist_mat = ssd.squareform(ssd.pdist(Xs))
	if floor(alpha) - alpha == 0:
		return nan2zero((-1)**(alpha+1)*dist_mat*np.log(dist_mat))
	else:
		return nan2zero((-1)**(floor(alpha)+1)*dist_mat**(2*alpha))

def krig_mat_landmark2 (alpha, Xs, Ys):
	"""
	Returns the krig-basis matrix of a list of points
	when there are only landmark constraints
	"""
	dist_mat = ssd.cdist(Ys, Xs)
	if floor(alpha) - alpha == 0:
		return nan2zero((-1)**(alpha+1)*dist_mat*np.log(dist_mat))
	else:
		return nan2zero((-1)**(floor(alpha)+1)*dist_mat**(2*alpha))


def krig_kernel_mat(alpha, Xs, Epts, E1s):
	"""
	computes kriging kernel matrix
	"""
	assert Xs.shape[1] == Epts.shape[1]
	assert E1s.shape[0] == Epts.shape[0]
	n, d = Xs.shape
	m, _ = Epts.shape
	X = np.tile(Xs, (1,m)).reshape(n, m, 1, d)
	EX = np.tile(Epts, (n, 1)).reshape(n, m, 1, d)
	XEdiff = X-EX # [[[x_1 - e_1], ..., [x_1 - e_m]], ..., [[x_n - e_1], ..., [x_n - e_m]]]

	XE = np.tile(Epts, (1,n)).reshape(m,n,1,d)
	X = np.tile(Xs, (m, 1)).reshape(m, n, 1, d)
	EXdiff = XE-X # [[[ex_1 - x_1], ..., [ex_1 - x_n]], ..., [[ex_m - x_1], ..., [ex_m - x_m]]]
	
	E_xs = np.tile(Epts, (1,m)).reshape(m, m, 1, d) 
	E_ys = np.tile(Epts, (m,1)).reshape(m, m, 1, d)
	Ediff = E_xs - E_ys #[[e1-e1], ..., [ e_m - e_1], ..., [e_m-e_1], ..., [e_m - e_m]]
	Exs = np.array([E1s[:, 0]]) #x components of E1s
	Eys = np.array([E1s[:, 1]]) #y components of E1s
	
	nX = np.sum(np.abs(XEdiff)**2,axis=-1).reshape(n,m) # squared norm of each element x_i-e_j of X-EX #can rewrite these with nlg.norm
	nEX = np.sum(np.abs(EXdiff)**2,axis=-1).reshape(m,n) # squared norm of each element ey_i-x_j of Y-EX

	dist_mat = ssd.squareform(ssd.pdist(Xs))**2 #squared norm between Xs
	E_dist_mat = ssd.squareform(ssd.pdist(Epts))**2 #squared norm between Epts

	xedist = XEdiff[0:, 0:, 0, 0] # difference in x coordinates of x_i and e_j i.e. x_i_x - e_j_x
	Exdist = Ediff[0:, 0:, 0, 0] # difference in x coordinates of e_i and e_j i.e. e_i_x - e_j_x
	exdist_x = EXdiff[0:, 0:, 0, 0] # difference in x coordinates of e_i and x_j i.e. ey_i_x - x_j_x
	yedist = XEdiff[0:, 0:, 0, 1] # difference in y coordinates of x_i and e_j i.e. x_i_x - e_j_x
	Eydist = Ediff[0:, 0:, 0, 1] # difference in y coordinates of e_i and e_j i.e. e_i_x - e_j_x
	exdist_y = EXdiff[0:, 0:, 0, 1] # difference in y coordinates of e_i and x_j i.e. ey_i_x - x_j_x

	if d == 2:	
		dist_mat_sqrt = np.sqrt(dist_mat)
		S_00   = dist_mat*dist_mat_sqrt #sigma
		
		nXsqrt = np.sqrt(nX)
		S_01x  = xedist*nXsqrt # dsigma/dx
		S_01y  = yedist*nXsqrt # dsigma/dy

		nEXsqrt = np.sqrt(nEX)
		S_10x  = exdist_x*nEXsqrt#dsigma/dx.T  #Transposes?
		S_10y  = exdist_y*nEXsqrt#dsigma/dy.T
		#S_10x  = Exs*S_01x.T#dsigma/dx.T
		#S_10y  = Eys*S_01y.T #dsigma/dy.T
		
		# is copying expensive?
		E_dist_mat1 = E_dist_mat.copy()
		np.fill_diagonal(E_dist_mat1, 1)
		Esqrt = np.sqrt(E_dist_mat)
		Esqrt1 = np.sqrt(E_dist_mat1)

		S_11xx = nan2zero(Esqrt + 2*(alpha-1)*np.square(Exdist)/Esqrt1) #dsigma/dxdy
		S_11xy = nan2zero(Exdist*Eydist/Esqrt1) #dsigma/dxdy
		S_11yy = nan2zero(Esqrt + 2*(alpha-1)*np.square(Eydist)/Esqrt1) #dsigma/dydy

		S_01   = Exs*S_01x + Eys*S_01y
		S_10   = Exs.T*S_10x + Eys.T*S_10y
		S_11   = -2*alpha*(Exs.T*Exs*S_11xx + Eys.T*Eys*S_11yy) -4*alpha*(Exs.T*Eys*S_11xy + Eys.T*Exs*S_11xy)

		return np.r_[np.c_[S_00, -2*alpha*S_01], np.c_[2*alpha*S_10, S_11]]

	elif d==3:
		zedist = XEdiff[0:, 0:, 0, 2] # difference in z coordinates of x_i and e_j i.e. x_i_x - e_j_x
		Ezdist = Ediff[0:, 0:, 0, 2] # difference in z coordinates of e_i and e_j i.e. e_i_x - e_j_x
		Ezs = np.array([E1s[:, 2]]) #z components of E1s
		exdist_z = EXdiff[0:, 0:, 0, 2]## difference in z coordinates of ey_i and x_j i.e. ey_i_x - x_j_x

		#top of Sigma
		dist_matsqrt = np.sqrt(dist_mat)
		S_00 = dist_mat*dist_matsqrt #sigma
		
		nXsqrt = np.sqrt(nX)
		S_01x = xedist*nXsqrt # dsigma/dx
		S_01y = yedist*nXsqrt # dsigma/dy
		S_01z = zedist*nXsqrt

		#bottom of Sigma
		nEXsqrt = np.sqrt(nEX)
		S_10x  = exdist_x*nEXsqrt#dsigma/dx.T
		S_10y  = exdist_y*nEXsqrt#dsigma/dy.T
		S_10z  = exdist_z*nEXsqrt#dsigma/dz.T
		
		E_dist_mat1 = E_dist_mat.copy()
		np.fill_diagonal(E_dist_mat1, 1)
		Esqrt = np.sqrt(E_dist_mat)
		Esqrt1 = np.sqrt(E_dist_mat1)
		
		S_11xx = nan2zero(Esqrt + 2*(alpha-1)*np.square(Exdist)/Esqrt1) #dsigma/dxdy
		S_11yy = nan2zero(Esqrt + 2*(alpha-1)*np.square(Eydist)/Esqrt1) #dsigma/dydy
		S_11zz = nan2zero(Esqrt + 2*(alpha-1)*np.square(Ezdist)/Esqrt1) #dsigma/dzdz
		S_11xy = nan2zero(Exdist*Eydist/Esqrt1) #dsigma/dxdy
		S_11yz = nan2zero(Eydist*Ezdist/Esqrt1) #dsigma/dydz
		S_11xz = nan2zero(Exdist*Ezdist/Esqrt1) #dsigma/dxdz

		S_01   = Exs*S_01x + Eys*S_01y + Ezs*S_01z
		S_10   = Exs.T*S_10x + Eys.T*S_10y + Ezs.T*S_10z
		S_11   = -2*alpha*(Exs.T*Exs*S_11xx + Eys.T*Eys*S_11yy + Ezs.T*Ezs*S_11zz) - 4*alpha*(Exs.T*Eys*S_11xy + Eys.T*Exs*S_11xy + Eys.T*Ezs*S_11yz + Ezs.T*Eys*S_11yz+ Exs.T*Ezs*S_11xz + Ezs.T*Exs*S_11xz) 

		return np.r_[np.c_[S_00, -2*alpha*S_01], np.c_[2*alpha*S_10, S_11]]
	else:
		raise NotImplementedError

def krig_kernel_mat2(alpha, Xs, Epts, E1s, E1sr, Ys, Eypts):
	"""
	computes kriging kernel matrix
	Xs are centers of basis functions
	y_na are landmark points at which matrix is computed
	ey_na are normal points at which matrix is computed
	E1s are original normal lengths
	E1sr are new normal lengths
	"""
	assert Xs.shape[1] == Epts.shape[1]
	assert E1s.shape[0] == Epts.shape[0]
	n, d = Xs.shape
	m, _ = Epts.shape
	s, _ = Ys.shape
	j, _ = Eypts.shape
	Y = np.tile(Ys, (1,m)).reshape(s,m,1,d)
	EX = np.tile(Epts, (s, 1)).reshape(s, m, 1, d)
	YEdiff = Y-EX # [[[y_1 - e_1], ..., [y_1 - e_m]], ..., [[y_n - e_1], ..., [y_n - e_m]]]

	EY = np.tile(Eypts, (1,n)).reshape(j,n,1,d)
	X = np.tile(Xs, (j, 1)).reshape(j, n, 1, d)
	EYdiff = EY-X # [[[ey_1 - x_1], ..., [ey_1 - x_n]], ..., [[ey_m - x_1], ..., [ey_m - x_m]]]
	
	E_xs = np.tile(Eypts, (1,m)).reshape(j, m, 1, d) 
	E_ys = np.tile(Epts, (j,1)).reshape(j, m, 1, d)
	Ediff = E_xs - E_ys #[[ey1-e1], ..., [ey_1 - e_m], ..., [ey_m-e_1], ..., [ey_m - e_m]]
	
	Exs = np.array([E1s[:, 0]]) #x components of E1s
	Eys = np.array([E1s[:, 1]]) #y components of E1s
	Exsr = np.array([E1sr[:, 0]]) 
	Eysr = np.array([E1sr[:, 1]])
	
	
	nYE = np.sum(np.abs(YEdiff)**2,axis=-1).reshape(s,m) # squared norm of each element y_i-e_j of Y-EX
	nEY = np.sum(np.abs(EYdiff)**2,axis=-1).reshape(j,n) # squared norm of each element ey_i-x_j of Y-EX #checked
	
	dist_mat = ssd.cdist(Ys, Xs)**2 #squared norm between Xs and Ys
	E_dist_mat = ssd.cdist(Eypts, Epts)**2  #squared norm between Epts 

	yedist_x = YEdiff[0:, 0:, 0, 0] # difference in x coordinates of y_i and e_j i.e. x_i_x - e_j_x
	Edist_x = Ediff[0:, 0:, 0, 0] # difference in x coordinates of ey_i and e_j i.e. ey_i_x - e_j_x
	eydist_x = EYdiff[0:, 0:, 0, 0] # difference in x coordinates of ey_i and x_j i.e. ey_i_x - x_j_x
	yedist_y = YEdiff[0:, 0:, 0, 1] # difference in y coordinates of y_i and e_j i.e. x_i_x - e_j_x
	Edist_y = Ediff[0:, 0:, 0, 1] # difference in y coordinates of ey_i and e_j i.e. ey_i_x - e_j_x
	eydist_y = EYdiff[0:, 0:, 0, 1] # difference in y coordinates of ey_i and x_j i.e. ey_i_x - x_j_x

	if d == 2:	
		dist_mat_sqrt = np.sqrt(dist_mat)
		S_00   = dist_mat*dist_mat_sqrt
		
		nYEsqrt = np.sqrt(nYE)
		S_01x  = yedist_x*nYEsqrt #dsigma/dx
		S_01y  = yedist_y*nYEsqrt #dsigma/dy

		nEYsqrt = np.sqrt(nEY)
		S_10x  = eydist_x*nEYsqrt#dsigma/dx.T
		S_10y  = eydist_y*nEYsqrt#dsigma/dy.T
		
		Esqrt = np.sqrt(E_dist_mat)
		S_11xx = Esqrt + 2*(alpha-1)*nan2zero(np.square(Edist_x)/Esqrt) #dsigma/dxdy
		S_11xy = nan2zero(Edist_x*Edist_y/Esqrt) #dsigma/dxdy
		S_11yy = Esqrt + 2*(alpha-1)*nan2zero(np.square(Edist_y)/Esqrt) #dsigma/dydy

		S_01   = Exs*S_01x + Eys*S_01y
		S_10   = Exsr.T*S_10x + Eysr.T*S_10y
		S_11   = -2*alpha*(Exsr.T*Exs*S_11xx + Eysr.T*Eys*S_11yy) - 4*alpha*(Exsr.T*Eys*S_11xy + Eysr.T*Exs*S_11xy) 

		return np.r_[np.c_[S_00, -2*alpha*S_01], np.c_[2*alpha*S_10, S_11]]

	elif d==3:
		yedist_z = YEdiff[0:, 0:, 0, 2] # difference in z coordinates of x_i and e_j i.e. x_i_x - e_j_x
		Edist_z = Ediff[0:, 0:, 0, 2] # difference in z coordinates of e_i and e_j i.e. e_i_x - e_j_x
		Ezs = np.array([E1s[:, 2]]) #z components of E1s
		Ezsr = np.array([E1sr[:, 2]])
		eydist_z = EYdiff[0:, 0:, 0, 2]## difference in z coordinates of ey_i and x_j i.e. ey_i_x - x_j_x

		#top of Sigma
		dist_matsqrt = np.sqrt(dist_mat)
		S_00 = dist_mat*dist_matsqrt #sigma
		
		nYEsqrt = np.sqrt(nYE)
		S_01x = yedist_x*nYEsqrt # dsigma/dx
		S_01y = yedist_y*nYEsqrt # dsigma/dy
		S_01z = yedist_z*nYEsqrt

		#bottom of Sigma
		nEYsqrt = np.sqrt(nEY)
		S_10x  = eydist_x*nEYsqrt#dsigma/dx.T
		S_10y  = eydist_y*nEYsqrt#dsigma/dy.T
		S_10z  = eydist_z*nEYsqrt#dsigma/dz.T
		
		Esqrt = np.sqrt(E_dist_mat)
		S_11xx = Esqrt + 2*(alpha-1)*nan2zero(np.square(Edist_x)/Esqrt) #dsigma/dxdy1
		S_11yy = Esqrt + 2*(alpha-1)*nan2zero(np.square(Edist_y)/Esqrt) #dsigma/dydy
		S_11zz = Esqrt + 2*(alpha-1)*nan2zero(np.square(Edist_z)/Esqrt) #dsigma/dzdz
		S_11xy = nan2zero(Edist_x*Edist_y/Esqrt) #dsigma/dxdy
		S_11yz = nan2zero(Edist_y*Edist_z/Esqrt) #dsigma/dydz
		S_11xz = nan2zero(Edist_x*Edist_z/Esqrt) #dsigma/dxdz

		S_01   = Exs*S_01x + Eys*S_01y + Ezs*S_01z
		S_10   = Exsr.T*S_10x + Eysr.T*S_10y + Ezsr.T*S_10z
		#S_10   = Exsr*S_10x + Eysr*S_10y + Ezsr*S_10z
		S_11   = -2*alpha*(Exsr.T*Exs*S_11xx + Eysr.T*Eys*S_11yy + Ezsr.T*Ezs*S_11zz) - 4*alpha*(Exsr.T*Eys*S_11xy + Eysr.T*Exs*S_11xy + Eysr.T*Ezs*S_11yz + Ezsr.T*Eys*S_11yz+ Exsr.T*Ezs*S_11xz + Ezsr.T*Exs*S_11xz) 

		return np.r_[np.c_[S_00, -2*alpha*S_01], np.c_[2*alpha*S_10, S_11]]
	else:
		raise NotImplementedError

def krig_mat_linear(Xs, Epts, Exs):
	n, d = Xs.shape
	m,_ = Epts.shape
	assert Xs.shape[1] == Epts.shape[1]

	D1 = np.c_[np.ones((n,1)), Xs]
	D2 = np.c_[np.zeros((m,1)), Exs]

	return np.r_[D1, D2]
	#return np.r_[D1, D2]

def krig_fn(alpha, Xs, Ys, Epts, Exs, Eys):
	"""
	Computes the kriging function that maps Xs onto Ys
	and at points Epts1 maps normals Exs onto Eys
	"""
	assert Xs.shape[1] == Exs.shape[1]
	n, dim = Xs.shape
	m, _ = Exs.shape

	S = krig_kernel_mat(alpha, Xs, Epts, Exs)
	D = krig_mat_linear(Xs, Epts, Exs)
	K = np.r_[np.c_[S, D], np.c_[D.T, np.zeros((dim+1, dim+1))]]

	targ = np.r_[Ys, Eys, np.zeros((dim+1, dim))]

	return nlg.solve(K, targ)


def bending_energynormal(S,D, dim):
	
	#return np.r_[np.c_[S, D], np.c_[D.T, np.zeros((dim+1, dim+1))]]
	return S

def solve_eqp1_interest(H, f, A, b):
    """solve equality-constrained qp
    min tr(x'Hx) + sum(f'x)
    s.t. Ax = b
    """    

    n_vars = H.shape[0]
    assert H.shape[1] == n_vars
    assert f.shape[0] == n_vars
    assert A.shape[1] == n_vars
    n_cnts = A.shape[0]
    
    import IPython
    #IPython.embed()
    _u,_s,_vh = np.linalg.svd(A.T)
    N = _u[:,n_cnts:]
    P = nlg.pinv(A)
    Pb = P.dot(b)
    # columns of N span the null space

    nf = -(f + H.dot(Pb))
    
    # x = Nz
    # then problem becomes unconstrained minimization .5*z'NHNz + z'Nf
    # NHNz + Nf = 0
    z = np.linalg.solve(N.T.dot(H.dot(N)), -N.T.dot(f) - N.T.dot(H).dot(Pb))
    x = P.dot(b) + N.dot(z)
    
    return x

def krig_fit_interest(Xs, Ys, Epts, Exs, Eys, bend_coef = .01, normal_coef = 1, wt_n = None, rot_coefs = 1e-5, interest_pts_inds = None):
	assert Xs.shape[1] == Exs.shape[1]
	assert Xs.shape[1] == Ys.shape[1]
	assert Exs.shape[0] == Eys.shape[0]

	alpha = 1.5
	n, dim = Xs.shape

	m = Exs.shape[0]
	if wt_n is None: wt_n = np.ones(n+m)/(n+m)
	wt_n[-m:]*=normal_coef
	#rot_coefs = np.ones(dim) * rot_coefs if np.isscalar(rot_coefs) else rot_coefs

	Y = np.r_[Ys, Eys]

	S = krig_kernel_mat(alpha, Xs, Epts, Exs)
	D = krig_mat_linear(Xs, Epts, Exs)
	#delete interest_pts_inds from S, D, and B (or maybe not B)

	Q = np.c_[S, D]
	WQ = wt_n[:,None]*Q
	H = Q.T.dot(WQ)
	H[:n+m, :n+m] += bend_coef*S
	#H[n+m+1:, n+m+1:] += np.diag(rot_coefs) #wrong
	f = -WQ.T.dot(Y)
	#f[n+m+1:,0:dim] -= np.diag(rot_coefs) #check this #wrong

	interest_pts_inds1 = interest_pts_inds[:n]
	interest_pts_inds2 = interest_pts_inds[n:]

	if interest_pts_inds is not None:
		A = np.r_[np.c_[D.T, np.zeros((dim+1, dim+1))], np.c_[S[interest_pts_inds], D[interest_pts_inds]]]
		import IPython
		#IPython.embed()
		b = np.r_[np.zeros((dim+1, dim)), np.r_[Ys[interest_pts_inds1], Eys[interest_pts_inds2]]]
	else:
		A = np.c_[D.T, np.zeros((dim+1, dim+1))]
		b = np.zeros((dim+1, dim))
	
	#import IPython
	#IPython.embed()
	
	Theta = solve_eqp1_interest(H, f, A, b)
	
	return Theta[:n+m], Theta[n+m], Theta[n+m+1:]
	#return Theta

def krig_fit1Normal(alpha, Xs, Ys, Epts, Exs, Eys, bend_coef = .01, normal_coef = 1, wt_n = None, rot_coefs = 1e-5):
	# This uses euclidean difference for normals. Change it to angle difference?

	assert Xs.shape[1] == Exs.shape[1]
	assert Xs.shape[1] == Ys.shape[1]
	assert Exs.shape[0] == Eys.shape[0]
	
	n,dim = Xs.shape
	d = dim
	m,_ = Exs.shape
	if normal_coef != 0:
		if wt_n is None: wt_n = np.ones(n+m)/(n+m)
		wt_n[n:]*=normal_coef

		Y = np.r_[Ys, Eys]

		S = krig_kernel_mat(alpha, Xs, Epts, Exs)
		D = krig_mat_linear(Xs, Epts, Exs)
		B = bending_energynormal(S, D, dim)

		Q = np.c_[S, D]
		WQ = wt_n[:,None]*Q
		H = Q.T.dot(WQ)
		H[:n+m, :n+m] += bend_coef*B

		f = -WQ.T.dot(Y)

		A = np.c_[D.T, np.zeros((dim+1, dim+1))]

		Theta = solve_eqp1(H, f, A)

		return Theta[:n+m], Theta[n+m], Theta[n+m+1:]
	else:
		if wt_n is None: wt_n = np.ones(n)
		rot_coefs = np.ones(dim) * rot_coefs if np.isscalar(rot_coefs) else rot_coefs


		Y = Ys

		S = krig_mat_landmark(alpha, Xs)
		D = np.c_[np.ones((n,1)), Xs]
		B = bending_energynormal(S,D, dim)

		Q = np.c_[D,S]
		WQ = wt_n[:,None]*Q
		H = Q.T.dot(WQ)
		H[d+1:, d+1:] += bend_coef*B
		H[1:d+1, 1:d+1] += np.diag(rot_coefs)
		
		f = -WQ.T.dot(Y)
		f[1:d+1,0:d] -= np.diag(rot_coefs)

		A = np.r_[np.zeros((d+1,d+1)), D].T

		Theta = solve_eqp1(H, f, A)

		return Theta[d+1:], Theta[0], Theta[1:d+1]

def krig_fit3_landmark(alpha, Xs, Ys, bend_coef = .1, wt_n = None):
	n, dim = Xs.shape
	if wt_n is None: wt_n = np.ones(n)

	Y = Ys

	S = krig_mat_landmark(alpha, Xs)
	D = np.c_[np.ones((n,1)), Xs]
	B = bending_energynormal(S,D)

	Q = np.c_[S,D]
	WQ = wt_n[:,None]*Q
	H = Q.T.dot(WQ)
	H[:n, :n] += bend_coef*B
	f = -WQ.T.dot(Y)

	A = np.c_[D.T, np.zeros((dim+1, dim+1))]

	Theta = solve_eqp1(H, f, A)

	return Theta[:n], Theta[n], Theta[n+1:]

#Doesn't matter

def krig_objective2(Xs, Ys, Epts, Exs, Eys, bend_coef, alpha = 1.5, normal_coef = 1):
	"""
	Theta is coefficients of function
	Xs are source points
	Ys are target points
	B is bending energy matrix
	"""
	n, dim = Xs.shape
	m, _ = Exs.shape
	if wt_n is None: wt_n = np.ones(n+m)/(n+m)
	wt_n[n:]*=normal_coef
	#if wt_n is None: wt_n = np.ones(n+m)

	Y = np.r_[Ys, Eys]

	S = krig_kernel_mat(alpha, Xs, Epts, Exs)
	D = krig_mat_linear2(Xs, Epts)
	B = bending_energy(S, D)

	Q = np.c_[S, D]
	#WQ = wt_n[:, None]*Q
	WQ = wt_n[:,None]*Q
	H = Q.T.dot(WQ)
	H[:n+m, :n+m] += bend_coef*B
	f = -WQ.T.dot(Y)

	A = np.c_[D.T, np.zeros((6, 6))]

	Theta = solve_eqp1(H, f, A)

	targ = np.r_[Ys, Eys, np.zeros((6, dim))]

	K = np.r_[np.c_[S, D], np.c_[D.T, np.zeros((6, 6))]]
	
	point_diff = K.dot(Theta) - targ
	point_cost = np.sum(np.sum(np.abs(point_diff)**2, axis=-1)**.5)
	ba = Theta[:n+m]
	bend_cost = bend_coef*np.trace(ba.T.dot(B).dot(ba))

	return point_cost + bend_cost


def plot_objective(bend_coefs):
	cost = []
	for bend_coef in bend_coefs:
		cost.append(krig_objective(Xs, Ys, Epts, Exs, Eys, bend_coef))
	return cost




def krig_test(n_iter, alpha, Xs, Ys, Epts, Exs, Eys, bend_coef = .1, normal_coef = 1, wt_n = None):
	n,dim = Xs.shape
	m,_ = Exs.shape
	#wt_n = 1.0/(n+m)
	if wt_n is None: wt_n = np.ones(n+m)/(n+m)
	wt_n[n:]*=normal_coef
	S = krig_kernel_mat(alpha, Xs, Epts, Exs)
	D1 = np.c_[np.ones((n,1)), Xs]
	D2 = np.c_[np.zeros((m,1)), np.ones((m,dim))]
	D = np.r_[D1, D2]

	K = np.r_[np.c_[S, D], np.c_[D.T, np.zeros((dim+1, dim+1))]]
	for i in range(n_iter-1):
		Y = np.r_[Ys, Eys]

		S = krig_kernel_mat(alpha, Xs, Epts, Exs)
		D1 = np.c_[np.ones((n,1)), Xs]
		D2 = np.c_[np.zeros((m,1)), np.ones((m,dim))]
		D = np.r_[D1, D2]
		B = bending_energy(S, D)

		Q = np.c_[S, D]
		#WQ = wt_n[:, None]*Q
		WQ = wt_n[:,None]*Q
		H = Q.T.dot(WQ)
		H[:n+m, :n+m] += bend_coef*B
		f = -WQ.T.dot(Y)

		A = np.c_[D.T, np.zeros((dim+1, dim+1))]

		Theta = solve_eqp1(H, f, A)

		print np.abs(np.sum(K.dot(Theta) - np.r_[Ys, Eys, np.zeros((3,2))]))

		Ys = K.dot(Theta)[:n, :]
		Eys = K.dot(Theta)[n:n+m,:]

		
	
	Y = np.r_[Ys, Eys]

	S = krig_kernel_mat(alpha, Xs, Epts, Exs)
	D1 = np.c_[np.ones((n,1)), Xs]
	D2 = np.c_[np.zeros((m,1)), np.ones((m,dim))]
	D = np.r_[D1, D2]
	B = bending_energy(S, D)

	Q = np.c_[S, D]
	#WQ = wt_n[:, None]*Q
	WQ = wt_n[:,None]*Q
	H = Q.T.dot(WQ)
	H[:n+m, :n+m] += bend_coef*B
	f = -WQ.T.dot(Y)

	A = np.c_[D.T, np.zeros((dim+1, dim+1))]

	Theta = solve_eqp1(H, f, A)
	return Theta, K.dot(Theta) - np.r_[Ys, Eys, np.zeros((3,2))]



def krig_fit2t(alpha, Xs, Ys, Epts, Exs, Eys, bend_coef = .1, normal_coef = 1, wt_n = None):
	n,dim = Xs.shape
	m,_ = Exs.shape
	#wt_n = 1.0/(n+m)
	if wt_n is None: wt_n = np.ones(n+m)/(n+m)
	wt_n[n:]*=normal_coef
	#if wt_n is None: wt_n = np.ones(n+m)

	Y = np.r_[Ys, Eys]

	S = krig_kernel_mat(alpha, Xs, Epts, Exs)
	D = krig_mat_linear2(Xs, Epts)
	B1 = bending_energy(S, D)
	SD = np.c_[S,D]
	B = SD.T.dot(B1).dot(SD)

	Q = np.c_[S, D]
	#WQ = wt_n[:, None]*Q
	WQ = wt_n[:,None]*Q
	H = Q.T.dot(WQ)
	#H[:n+m, :n+m] += bend_coef*B
	H += bend_coef*B
	f = -WQ.T.dot(Y)

	A = np.c_[D.T, np.zeros((6, 6))]

	Theta = solve_eqp1(H, f, A)


	K = np.r_[np.c_[S, D], np.c_[D.T, np.zeros((6,6))]]

	targ = np.r_[Ys, Eys, np.zeros((6, dim))]

	return Theta[:n+m], Theta[n+m:]



#derivative testing

def derivativex(f, x, epsilon = 1e-6):
	xpert = x.copy()
	xpert[:,0] = xpert[:,0] + epsilon
	return (f(xpert) - f(x))/epsilon

def derivativey(f, x, epsilon = 1e-6):
	xpert = x.copy()
	xpert[:,1] = xpert[:,1] + epsilon
	return (f(xpert)-f(x))/epsilon

def derivativez(f, x, epsilon = 1e-6):
	xpert = x.copy()
	xpert[:,2] += epsilon
	return (f(xpert) - f(x))/epsilon

def derivativexx(f, x, epsilon = 1e-6):
	xpert = x.copy()
	xpert[:,0] = xpert[:,0] + epsilon
	return (derivativex(f, xpert, epsilon) - derivativex(f,x,epsilon))/epsilon

def derivativexy(f, x, epsilon = 1e-6):
	xpert = x.copy()
	xpert[:,1] = xpert[:,1] + epsilon
	return (derivativex(f, xpert, epsilon) - derivativex(f,x,epsilon))/epsilon

def derivativeyx(f, x, epsilon = 1e-6):
	xpert = x.copy()
	xpert[:,0] = xpert[:,0] + epsilon
	return (derivativey(f, xpert, epsilon) - derivativey(f,x,epsilon))/epsilon

def derivativeyy(f, x, epsilon = 1e-6):
	xpert = x.copy()
	xpert[:,1] = xpert[:,1] + epsilon
	return (derivativey(f, xpert, epsilon) - derivativey(f,x,epsilon))/epsilon

#normals finding

def find_rope_normals(pts1):
	normals = find_all_normals_naive(pts1, .15, flip_away=True)
	avg = np.average(pts1)
	for ex in normals:
		if ex[2] < avg:
			ex[2] = -ex[2]

	return normals

def find_rope_tangents(pts1):
	n, d = pts1.shape
	tangents_2d = np.zeros((n, d-1))
	pts1_2d = pts1[:,:2]
	for i in xrange(n-1):
		tangents_2d[i] = pts1_2d[i+1] - pts1_2d[i-1]
	#tangents_2d /= nlg.norm(tangents_2d, 1)
	tangents = np.c_[tangents_2d, np.zeros((n,1))]
	tangents[n-1] = tangents[n-2]
	tangents[0] = tangents[1]
	tangents /= nlg.norm(tangents, axis=1)[:,None]
	return tangents



"""

n,dim = Xs.shape
m,d  = Exs.shape

X = np.tile(Xs, (1,n)).reshape(n, n, 1, d)
XX = np.tile(Xs, (n, 1)).reshape(n, n, 1, d)
XXdiff = X-XX # [
xxdist = XXdiff[0:, 0:, 0, 0]
yxdist = XXdiff[0:, 0:, 0, 1]
dist_mat = ssd.squareform(ssd.pdist(Xs))**2 #squared norm between Xs

X = np.tile(Xs, (1,m)).reshape(n, m, 1, d)
EX = np.tile(Epts, (n, 1)).reshape(n, m, 1, d)
XEdiff = X-EX # [[[x_1 - e_1], ..., [x_1 - e_m]], ..., [[x_n - e_1], ..., [x_n - e_m]]]
nX = np.sum(np.abs(XEdiff)**2,axis=-1).reshape(n,m)
xedist = XEdiff[0:, 0:, 0, 0] # difference in x coordinates of x_i and e_j i.e. x_i_x - e_j_x
yedist = XEdiff[0:, 0:, 0, 1] # difference in y coordinates of x_i and e_j i.e. x_i_x - e_j_x

S_00   = (-1)**(floor(alpha)+1)*dist_mat**(alpha) #sigma

S_01x  = -2*alpha*(-1)**(floor(alpha)+1)*xedist*nX**(alpha-1)
S_01y  = -2*alpha*(-1)**(floor(alpha)+1)*yedist*nX**(alpha-1)

S_10x  = 2*alpha*(-1)**(floor(alpha)+1)*xxdist*dist_mat**(alpha-1)
S_10y  = 2*alpha*(-1)**(floor(alpha)+1)*yxdist*dist_mat**(alpha-1)

S_11xx = -2*alpha*(-1)**(floor(alpha)+1)*(nX**(alpha-1) + 2*(alpha-1)*nan2zero(xedist**2*nX**(alpha-2))) #dsigma/dxdy
S_11xy = -4*alpha*(alpha-1)*(-1)**(floor(alpha)+1)*nan2zero(xedist*yedist*nX**(alpha-2)) #dsigma/dxdy
S_11yy = -2*alpha*(-1)**(floor(alpha)+1)*(nX**(alpha-1) + 2*(alpha-1)*nan2zero(yedist**2*nX**(alpha-2))) #dsigma/dydy


S_11x  = (S_11xx + S_11xy)
S_11y  = (S_11xy + S_11yy)

S_01   = S_01x + S_01y
S_10   = S_10x + S_10y
S_11   = S_11x + S_11y


D1 = np.c_[np.ones((n,1)), Xs, Xs[:,0][:,None]**2, Xs[:,0][:,None]*Xs[:,1][:,None], Xs[:, 1][:,None]**2]
D2 = np.c_[np.zeros((m,1)), np.ones((m,d)), 2*Epts[:,0][:,None], Epts[:,0][:,None] + Epts[:,1][:,None], 2*Epts[:,1][:,None] ]

S1 = np.c_[S_00, S_01]
S2 = np.c_[S_10, S_11]


dist_mat = ssd.cdist(pts, pts1)**2

assert Xs.shape[1] == Exs.shape[1]
assert Xs.shape[1] == Ys.shape[1]

n,dim = Xs.shape
m,_ = Exs.shape
if wt_n is None: wt_n = np.ones(n+m)/(n+m)
wt_n[n:]*=normal_coef
rot_coefs = np.ones(dim) * rot_coefs if np.isscalar(rot_coefs) else rot_coefs

Y = np.r_[Ys, Eys]

S = krig_kernel_mat(alpha, Xs, Epts, Exs)
D = krig_mat_linear(Xs, Epts, Exs)
B = bending_energynormal(S, D, dim)

Q = np.c_[S, D]
WQ = wt_n[:,None]*Q
H = Q.T.dot(WQ)
H[:n+m, :n+m] += bend_coef*B
#H[n+m+1:, n+m+1:] += np.diag(rot_coefs)
f = -WQ.T.dot(Y)
#f[n+m+1:,0:dim] -= np.diag(rot_coefs) #check this

A = np.c_[D.T, np.zeros((dim+1, dim+1))]

Theta = solve_eqp1(H, f, A)



"""





def main():
    from tn_testing.test_tps import gen_half_sphere, gen_half_sphere_pulled_in
    from tn_eval.tps_utils import find_all_normals_naive
    pts1 = gen_half_sphere(1, 30)
    pts2 = gen_half_sphere_pulled_in(1, 30, 4, .2)
    e1 = find_all_normals_naive(pts1, .7, flip_away=True)
    e2 = find_all_normals_naive(pts2, .7, flip_away=True)
    Xs = pts1
    Epts = pts1
    Exs = e1
    Ys = pts2
    Eys = e2
    alpha = 1.5

    krig_fit1Normal(alpha, Xs, Ys, Epts, Exs, Eys)





if __name__ == "__main__":
    main()


"""
n, d = Xs.shape
m, _ = Epts.shape
X = np.tile(Xs, (1,m)).reshape(n, m, 1, d)
EX = np.tile(Epts, (n, 1)).reshape(n, m, 1, d)
XEdiff = X-EX # [[[x_1 - e_1], ..., [x_1 - e_m]], ..., [[x_n - e_1], ..., [x_n - e_m]]]

XE = np.tile(Epts, (1,n)).reshape(m,n,1,d)
X = np.tile(Xs, (m, 1)).reshape(m, n, 1, d)
EXdiff = XE-X # [[[ex_1 - x_1], ..., [ex_1 - x_n]], ..., [[ex_m - x_1], ..., [ex_m - x_m]]]
	
E_xs = np.tile(Epts, (1,m)).reshape(m, m, 1, d) 
E_ys = np.tile(Epts, (m,1)).reshape(m, m, 1, d)
Ediff = E_xs - E_ys #[[e1-e1], ..., [ e_m - e_1], ..., [e_m-e_1], ..., [e_m - e_m]]
Exs = np.array([E1s[:, 0]]) #x components of E1s
Eys = np.array([E1s[:, 1]]) #y components of E1s
	
nX = np.sum(np.abs(XEdiff)**2,axis=-1).reshape(n,m) # squared norm of each element x_i-e_j of X-EX #can rewrite these with nlg.norm
nEX = np.sum(np.abs(EXdiff)**2,axis=-1).reshape(m,n) # squared norm of each element ey_i-x_j of Y-EX

dist_mat = ssd.squareform(ssd.pdist(Xs, 'sqeuclidean')) #squared norm between Xs
E_dist_mat = ssd.squareform(ssd.pdist(Epts))**2 #squared norm between Epts

xedist = XEdiff[0:, 0:, 0, 0] # difference in x coordinates of x_i and e_j i.e. x_i_x - e_j_x
Exdist = Ediff[0:, 0:, 0, 0] # difference in x coordinates of e_i and e_j i.e. e_i_x - e_j_x
exdist_x = EXdiff[0:, 0:, 0, 0] # difference in x coordinates of e_i and x_j i.e. ey_i_x - x_j_x
yedist = XEdiff[0:, 0:, 0, 1] # difference in y coordinates of x_i and e_j i.e. x_i_x - e_j_x
Eydist = Ediff[0:, 0:, 0, 1] # difference in y coordinates of e_i and e_j i.e. e_i_x - e_j_x
exdist_y = EXdiff[0:, 0:, 0, 1] # difference in y coordinates of e_i and x_j i.e. ey_i_x - x_j_x

		
dist_mat_sqrt = np.sqrt(dist_mat)
S_00   = dist_mat*dist_mat_sqrt #sigma
		
nXsqrt = np.sqrt(nX)
S_01x  = Exs*xedist*nXsqrt # dsigma/dx
S_01y  = Eys*yedist*nXsqrt # dsigma/dy

nEXsqrt = np.sqrt(nEX)
S_10x  = 3*exdist_x*nEXsqrt#dsigma/dx.T  #Transposes?
S_10y  = 3*exdist_y*nEXsqrt#dsigma/dy.T

		
# is copying expensive?
E_dist_mat1 = E_dist_mat.copy()
np.fill_diagonal(E_dist_mat1, 1)
Esqrt = np.sqrt(E_dist_mat)
Esqrt1 = np.sqrt(E_dist_mat1)

S_11xx = Esqrt + 2*(alpha-1)*np.square(Exdist)/Esqrt1 #dsigma/dxdy
S_11xy = Exdist*Eydist/Esqrt1 #dsigma/dxdy
S_11yy = Esqrt + 2*(alpha-1)*np.square(Eydist)/Esqrt1 #dsigma/dydy

S_01   = -2*alpha*(S_01x + S_01y)
S_10   = 2*alpha*(S_10x + S_10y)
S_11   = -2*alpha*(Exs.T*Exs*S_11xx + Eys.T*Eys*S_11yy) -4*alpha*(Exs.T*Eys*S_11xy + Eys.T*Exs*S_11xy)





alpha = 1.5
_,dim = Xs.shape

wt_n = None

normal_coef = 1


n = Xs.shape[0] + Exs.shape[0]

m,_ =Exs.shape

if wt_n is None: wt_n = np.ones(n)/n
wt_n[-m:]*=normal_coef
	#rot_coefs = np.ones(dim) * rot_coefs if np.isscalar(rot_coefs) else rot_coefs

Y = np.r_[Ys, Eys]

S = krig_kernel_mat(alpha, Xs, Epts, Exs)
D = krig_mat_linear(Xs, Epts, Exs)
	#delete interest_pts_inds from S, D, and B (or maybe not B)

K = np.r_[np.c_[S,D], np.c_[D.T, np.zeros((dim+1,dim+1))]]
targ = np.r_[Ys, Eys, np.zeros((4,3))]

Q = np.c_[S, D]
WQ = wt_n[:,None]*Q
H = Q.T.dot(WQ)
H[:n, :n] += bend_coef*S
	#H[n+m+1:, n+m+1:] += np.diag(rot_coefs) #wrong
f = -WQ.T.dot(Y)
	#f[n+m+1:,0:dim] -= np.diag(rot_coefs) #check this #wrong

interest_pts_inds2 = np.r_[interest_pts_inds, interest_pts_inds]


A = np.r_[np.c_[D.T, np.zeros((dim+1, dim+1))], np.c_[S[interest_pts_inds2], D[interest_pts_inds2]]]
b = np.r_[np.zeros((dim+1, dim)), np.r_[Ys[interest_pts_inds], Eys[interest_pts_inds]]]
n_vars = H.shape[0]
assert H.shape[1] == n_vars
assert f.shape[0] == n_vars
assert A.shape[1] == n_vars
n_cnts = A.shape[0]
    
_u,_s,_vh = np.linalg.svd(A.T)
N = _u[:,n_cnts:]
P = nlg.pinv(A)

Pb = P.dot(b)
nf = f + Pb
    # columns of N span the null space
    
    # x = Nz
    # then problem becomes unconstrained minimization .5*z'NHNz + z'Nf
    # NHNz + Nf = 0
z1 = np.linalg.solve(N.T.dot(H.dot(N)), -N.T.dot(f))
z2 = np.linalg.solve(N.T.dot(H.dot(N)), -N.T.dot(f) - N.T.dot(H).dot(Pb))
x = P.dot(b) + N.dot(z)
	
Theta = solve_eqp1_interest(H, f, A, b)

"""


