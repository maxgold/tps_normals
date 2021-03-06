import numpy as np, numpy.linalg as nlg
import scipy as sp, scipy.spatial.distance as ssd
from math import floor

from tn_rapprentice.tps import solve_eqp1, nan2zero



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
	
	
	nX = np.sum(np.abs(XEdiff)**2,axis=-1).reshape(n,m) # squared norm of each element x_i-e_j of X-EX
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
		S_00   = (-1)**(floor(alpha)+1)*dist_mat**(alpha) #sigma
		S_01x  = Exs*-2*alpha*(-1)**(floor(alpha)+1)*xedist*nX**(alpha-1) # dsigma/dx
		S_01y  = Eys*-2*alpha*(-1)**(floor(alpha)+1)*yedist*nX**(alpha-1) # dsigma/dy

		S_10x  = Exs.T*2*alpha*(-1)**(floor(alpha)+1)*exdist_x*nEX**(alpha-1)#dsigma/dx.T
		S_10y  = Eys.T*2*alpha*(-1)**(floor(alpha)+1)*exdist_y*nEX**(alpha-1)#dsigma/dy.T
		#S_10x  = Exs*S_01x.T#dsigma/dx.T
		#S_10y  = Eys*S_01y.T #dsigma/dy.T
		
		S_11xx = -2*alpha*(-1)**(floor(alpha)+1)*(E_dist_mat**(alpha-1) + 2*(alpha-1)*nan2zero(Exdist**2*E_dist_mat**(alpha-2))) #dsigma/dxdy
		S_11xy = -4*alpha*(alpha-1)*(-1)**(floor(alpha)+1)*nan2zero(Exdist*Eydist*E_dist_mat**(alpha-2)) #dsigma/dxdy
		S_11yy = -2*alpha*(-1)**(floor(alpha)+1)*(E_dist_mat**(alpha-1) + 2*(alpha-1)*nan2zero(Eydist**2*E_dist_mat**(alpha-2))) #dsigma/dydy

		S_01   = S_01x + S_01y
		S_10   = S_10x + S_10y
		S_11   = Exs.T*Exs*S_11xx + Exs.T*Eys*S_11xy + Eys.T*Exs*S_11xy + Eys.T*Eys*S_11yy

		return np.r_[np.c_[S_00, S_01], np.c_[S_10, S_11]]

	elif d==3:
		zedist = XEdiff[0:, 0:, 0, 2] # difference in z coordinates of x_i and e_j i.e. x_i_x - e_j_x
		Ezdist = Ediff[0:, 0:, 0, 2] # difference in z coordinates of e_i and e_j i.e. e_i_x - e_j_x
		Ezs = np.array([E1s[:, 2]]) #z components of E1s
		exdist_z = EXdiff[0:, 0:, 0, 2]## difference in z coordinates of ey_i and x_j i.e. ey_i_x - x_j_x

		#top of Sigma
		S_00 = (-1)**(floor(alpha)+1)*dist_mat**(alpha) #sigma
		S_01x = Exs*-2*alpha*(-1)**(floor(alpha)+1)*xedist*nX**(alpha-1) # dsigma/dx
		S_01y = Eys*-2*alpha*(-1)**(floor(alpha)+1)*yedist*nX**(alpha-1) # dsigma/dy
		S_01z = Ezs*-2*alpha*(-1)**(floor(alpha)+1)*zedist*nX**(alpha-1)

		#bottom of Sigma
		S_10x  = Exs.T*2*alpha*(-1)**(floor(alpha)+1)*exdist_x*nEX**(alpha-1)#dsigma/dx.T
		S_10y  = Eys.T*2*alpha*(-1)**(floor(alpha)+1)*exdist_y*nEX**(alpha-1)#dsigma/dy.T
		S_10z  = Ezs.T*2*alpha*(-1)**(floor(alpha)+1)*exdist_z*nEX**(alpha-1)#dsigma/dz.T
		
		S_11xx = -2*alpha*(-1)**(floor(alpha)+1)*(E_dist_mat**(alpha-1) + 2*(alpha-1)*nan2zero(Exdist**2*E_dist_mat**(alpha-2))) #dsigma/dxdy
		S_11yy = -2*alpha*(-1)**(floor(alpha)+1)*(E_dist_mat**(alpha-1) + 2*(alpha-1)*nan2zero(Eydist**2*E_dist_mat**(alpha-2))) #dsigma/dydy
		S_11zz = -2*alpha*(-1)**(floor(alpha)+1)*(E_dist_mat**(alpha-1) + 2*(alpha-1)*nan2zero(Ezdist**2*E_dist_mat**(alpha-2))) #dsigma/dzdz
		S_11xy = -4*alpha*(alpha-1)*(-1)**(floor(alpha)+1)*nan2zero(Exdist*Eydist*E_dist_mat**(alpha-2)) #dsigma/dxdy
		S_11yz = -4*alpha*(alpha-1)*(-1)**(floor(alpha)+1)*nan2zero(Eydist*Ezdist*E_dist_mat**(alpha-2)) #dsigma/dydz
		S_11xz = -4*alpha*(alpha-1)*(-1)**(floor(alpha)+1)*nan2zero(Exdist*Ezdist*E_dist_mat**(alpha-2)) #dsigma/dxdz

		S_01   = S_01x + S_01y + S_01z
		S_10   = S_10x + S_10y + S_10z
		S_11   = Exs.T*Exs*S_11xx + Eys.T*Eys*S_11yy + Ezs.T*Ezs*S_11zz + Exs.T*Eys*S_11xy + Eys.T*Exs*S_11xy + Eys.T*Ezs*S_11yz + Ezs.T*Eys*S_11yz+ Exs.T*Ezs*S_11xz + Ezs.T*Exs*S_11xz 

		return np.r_[np.c_[S_00, S_01], np.c_[S_10, S_11]]
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
	n, d = Xs.shape
	m, _ = Epts.shape
	s, _ = Ys.shape
	Y = np.tile(Ys, (1,m)).reshape(s,m,1,d)
	EX = np.tile(Epts, (s, 1)).reshape(s, m, 1, d)
	YEdiff = Y-EX # [[[y_1 - e_1], ..., [y_1 - e_m]], ..., [[y_n - e_1], ..., [y_n - e_m]]]

	EY = np.tile(Eypts, (1,n)).reshape(m,n,1,d)
	X = np.tile(Xs, (m, 1)).reshape(m, n, 1, d)
	EYdiff = EY-X # [[[ey_1 - x_1], ..., [ey_1 - x_n]], ..., [[ey_m - x_1], ..., [ey_m - x_m]]]
	
	E_xs = np.tile(Eypts, (1,m)).reshape(m, m, 1, d) 
	E_ys = np.tile(Epts, (m,1)).reshape(m, m, 1, d)
	Ediff = E_xs - E_ys #[[ey1-e1], ..., [ey_1 - e_m], ..., [ey_m-e_1], ..., [ey_m - e_m]]
	
	Exs = np.array([E1s[:, 0]]) #x components of E1s
	Eys = np.array([E1s[:, 1]]) #y components of E1s
	Exsr = np.array([E1sr[:, 0]]) 
	Eysr = np.array([E1sr[:, 1]])
	
	
	nYE = np.sum(np.abs(YEdiff)**2,axis=-1).reshape(s,m) # squared norm of each element y_i-e_j of Y-EX
	nEY = np.sum(np.abs(EYdiff)**2,axis=-1).reshape(m,n) # squared norm of each element ey_i-x_j of Y-EX #checked
	
	dist_mat = ssd.cdist(Ys, Xs)**2 #squared norm between Xs and Ys
	E_dist_mat = ssd.cdist(Eypts, Epts)**2  #squared norm between Epts 

	yedist_x = YEdiff[0:, 0:, 0, 0] # difference in x coordinates of y_i and e_j i.e. x_i_x - e_j_x
	Edist_x = Ediff[0:, 0:, 0, 0] # difference in x coordinates of ey_i and e_j i.e. ey_i_x - e_j_x
	eydist_x = EYdiff[0:, 0:, 0, 0] # difference in x coordinates of ey_i and x_j i.e. ey_i_x - x_j_x
	yedist_y = YEdiff[0:, 0:, 0, 1] # difference in y coordinates of y_i and e_j i.e. x_i_x - e_j_x
	Edist_y = Ediff[0:, 0:, 0, 1] # difference in y coordinates of ey_i and e_j i.e. ey_i_x - e_j_x
	eydist_y = EYdiff[0:, 0:, 0, 1] # difference in y coordinates of ey_i and x_j i.e. ey_i_x - x_j_x

	if d == 2:	
		S_00   = (-1)**(floor(alpha)+1)*dist_mat**(alpha) #sigma
		
		S_01x  = -2*alpha*(-1)**(floor(alpha)+1)*yedist_x*nYE**(alpha-1) # dsigma/dx
		S_01y  = -2*alpha*(-1)**(floor(alpha)+1)*yedist_y*nYE**(alpha-1) # dsigma/dy

		S_10x  = 2*alpha*(-1)**(floor(alpha)+1)*eydist_x*nEY**(alpha-1)#dsigma/dx.T
		S_10y  = 2*alpha*(-1)**(floor(alpha)+1)*eydist_y*nEY**(alpha-1)#dsigma/dy.T
		
		S_11xx = -2*alpha*(-1)**(floor(alpha)+1)*(E_dist_mat**(alpha-1) + 2*(alpha-1)*nan2zero(Edist_x**2*E_dist_mat**(alpha-2))) #dsigma/dxdy
		S_11xy = -4*alpha*(alpha-1)*(-1)**(floor(alpha)+1)*nan2zero(Edist_x*Edist_y*E_dist_mat**(alpha-2)) #dsigma/dxdy
		S_11yy = -2*alpha*(-1)**(floor(alpha)+1)*(E_dist_mat**(alpha-1) + 2*(alpha-1)*nan2zero(Edist_y**2*E_dist_mat**(alpha-2))) #dsigma/dydy

		S_01   = Exs*S_01x + Eys*S_01y
		S_10   = Exsr.T*S_10x + Eysr.T*S_10y
		S_11   = Exsr.T*Exs*S_11xx + Exsr.T*Eys*S_11xy + Eysr.T*Exs*S_11xy + Eysr.T*Eys*S_11yy

		return np.r_[np.c_[S_00, S_01], np.c_[S_10, S_11]]

	elif d==3:
		#something is wrong with the Ezsr part
		yedist_z = YEdiff[0:, 0:, 0, 2] # difference in z coordinates of x_i and e_j i.e. x_i_x - e_j_x
		Edist_z = Ediff[0:, 0:, 0, 2] # difference in z coordinates of e_i and e_j i.e. e_i_x - e_j_x
		Ezs = np.array([E1s[:, 2]]) #z components of E1s
		Ezsr = np.array([E1sr[:, 2]])
		eydist_z = EYdiff[0:, 0:, 0, 2]## difference in z coordinates of ey_i and x_j i.e. ey_i_x - x_j_x

		#top of Sigma
		S_00 = (-1)**(floor(alpha)+1)*dist_mat**(alpha) #sigma
		
		S_01x = -2*alpha*(-1)**(floor(alpha)+1)*yedist_x*nYE**(alpha-1) # dsigma/dx
		S_01y = -2*alpha*(-1)**(floor(alpha)+1)*yedist_y*nYE**(alpha-1) # dsigma/dy
		S_01z = -2*alpha*(-1)**(floor(alpha)+1)*yedist_z*nYE**(alpha-1)

		#bottom of Sigma
		S_10x  = 2*alpha*(-1)**(floor(alpha)+1)*eydist_x*nEY**(alpha-1)#dsigma/dx.T
		S_10y  = 2*alpha*(-1)**(floor(alpha)+1)*eydist_y*nEY**(alpha-1)#dsigma/dy.T
		S_10z  = 2*alpha*(-1)**(floor(alpha)+1)*eydist_z*nEY**(alpha-1)#dsigma/dz.T
		
		S_11xx = -2*alpha*(-1)**(floor(alpha)+1)*(E_dist_mat**(alpha-1) + 2*(alpha-1)*nan2zero(Edist_x**2*E_dist_mat**(alpha-2))) #dsigma/dxdy
		S_11yy = -2*alpha*(-1)**(floor(alpha)+1)*(E_dist_mat**(alpha-1) + 2*(alpha-1)*nan2zero(Edist_y**2*E_dist_mat**(alpha-2))) #dsigma/dydy
		S_11zz = -2*alpha*(-1)**(floor(alpha)+1)*(E_dist_mat**(alpha-1) + 2*(alpha-1)*nan2zero(Edist_z**2*E_dist_mat**(alpha-2))) #dsigma/dzdz
		S_11xy = -4*alpha*(alpha-1)*(-1)**(floor(alpha)+1)*nan2zero(Edist_x*Edist_y*E_dist_mat**(alpha-2)) #dsigma/dxdy
		S_11yz = -4*alpha*(alpha-1)*(-1)**(floor(alpha)+1)*nan2zero(Edist_y*Edist_z*E_dist_mat**(alpha-2)) #dsigma/dydz
		S_11xz = -4*alpha*(alpha-1)*(-1)**(floor(alpha)+1)*nan2zero(Edist_x*Edist_z*E_dist_mat**(alpha-2)) #dsigma/dxdz

		S_01   = Exs*S_01x + Eys*S_01y + Ezs*S_01z
		S_10   = Exsr.T*S_10x + Eysr.T*S_10y + Ezsr.T*S_10z
		S_11   = Exsr.T*Exs*S_11xx + Eysr.T*Eys*S_11yy + Ezsr.T*Ezs*S_11zz + Exsr.T*Eys*S_11xy + Eysr.T*Exs*S_11xy + Eysr.T*Ezs*S_11yz + Ezsr.T*Eys*S_11yz+ Exsr.T*Ezs*S_11xz + Ezsr.T*Exs*S_11xz 

		return np.r_[np.c_[S_00, S_01], np.c_[S_10, S_11]]
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
def krig_mat_linear2(Xs, Epts):
	n, d = Xs.shape
	m,_ = Epts.shape
	assert Xs.shape[1] == Epts.shape[1]

	D1 = np.c_[np.ones((n,1)), Xs, Xs[:,0][:,None]**2, Xs[:,0][:,None]*Xs[:,1][:,None], Xs[:, 1][:,None]**2]
	D2 = np.c_[np.zeros((m,1)), np.ones((m,d)), 2*Epts[:,0][:,None], Epts[:,0][:,None] + Epts[:,1][:,None], 2*Epts[:,1][:,None] ]

	return np.r_[D1, D2]

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

def bending_energyweird(S, D, dim):
	"""
	computes bending energy of kriging function given by matrices S and D
	S is the kriging kernel matrix and D is the linear matrix
	"""
	"""
	DTDinv = nlg.inv(D.T.dot(D))
	P = D.dot(DTDinv).dot(D.T)
	I = np.eye(P.shape[0])
	B1 = (I-P).dot(S).dot(I-P)
	B = nlg.pinv(B1)
	"""


	K = np.r_[np.c_[S, D], np.c_[D.T, np.zeros((dim+1, dim+1))]]

	#return nlg.inv(K)
	return nlg.inv(K)

def bending_energynormal(S,D, dim):
	
	#return np.r_[np.c_[S, D], np.c_[D.T, np.zeros((dim+1, dim+1))]]
	return S

def krig_fit1Weird(alpha, Xs, Ys, Epts, Exs, Eys, bend_coef = .1, normal_coef = 1, wt_n = None):
	assert Xs.shape[1] == Exs.shape[1]
	assert Xs.shape[1] == Ys.shape[1]
	
	n,dim = Xs.shape
	m,_ = Exs.shape
	#wt_n = 1.0/(n+m)
	if wt_n is None: wt_n = np.ones(n+m)/(n+m)
	wt_n[n:]*=normal_coef
	#if wt_n is None: wt_n = np.ones(n+m)

	Y = np.r_[Ys, Eys]

	S = krig_kernel_mat(alpha, Xs, Epts, Exs)
	D = krig_mat_linear(Xs, Epts, Exs)
	B = bending_energyweird(S, D, dim)

	Q = np.c_[S, D]
	#WQ = wt_n[:, None]*Q
	WQ = wt_n[:,None]*Q
	H = Q.T.dot(WQ)
	#H[:n+m, :n+m] += bend_coef*B
	H[:n+m, :n+m] += bend_coef*B[:n+m, :n+m]
	f = -WQ.T.dot(Y)

	A = np.c_[D.T, np.zeros((dim+1, dim+1))]

	Theta = solve_eqp1(H, f, A)

	return Theta[:n+m], Theta[n+m:]

def krig_fit1Normal(alpha, Xs, Ys, Epts, Exs, Eys, bend_coef = .1, normal_coef = 1, wt_n = None):
	assert Xs.shape[1] == Exs.shape[1]
	assert Xs.shape[1] == Ys.shape[1]
	
	n,dim = Xs.shape
	m,_ = Exs.shape
	#wt_n = 1.0/(n+m)
	if wt_n is None: wt_n = np.ones(n+m)/(n+m)
	wt_n[n:]*=normal_coef
	#if wt_n is None: wt_n = np.ones(n+m)

	Y = np.r_[Ys, Eys]

	S = krig_kernel_mat(alpha, Xs, Epts, Exs)
	D = krig_mat_linear(Xs, Epts, Exs)
	B = bending_energynormal(S, D, dim)

	Q = np.c_[S, D]
	#WQ = wt_n[:, None]*Q
	WQ = wt_n[:,None]*Q
	H = Q.T.dot(WQ)
	#H[:n+m, :n+m] += bend_coef*B
	H[:n+m, :n+m] += bend_coef*B
	f = -WQ.T.dot(Y)

	A = np.c_[D.T, np.zeros((dim+1, dim+1))]

	Theta = solve_eqp1(H, f, A)

	return Theta[:n+m], Theta[n+m:]


def krig_fit2(alpha, Xs, Ys, Epts, Exs, Eys, bend_coef = .1, normal_coef = 1, wt_n = None):
	n,dim = Xs.shape
	m,_ = Exs.shape
	#wt_n = 1.0/(n+m)
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

	return Theta[:n+m], Theta[n+m:]

def krig_fit3_landmark(alpha, Xs, Ys, bend_coef = .1, wt_n = None):
	n, dim = Xs.shape
	if wt_n is None: wt_n = np.ones(n)

	Y = Ys

	S = krig_mat_landmark(alpha, Xs)
	D = np.c_[np.ones((n,1)), Xs]
	B = bending_energy(S,D)

	Q = np.c_[S,D]
	WQ = wt_n[:,None]*Q
	H = Q.T.dot(WQ)
	H[:n, :n] += bend_coef*B
	f = -WQ.T.dot(Y)

	A = np.c_[D.T, np.zeros((dim+1, dim+1))]

	Theta = solve_eqp1(H, f, A)

	return Theta[:n], Theta[n:]


def krig_objective(Xs, Ys, Epts, Exs, Eys, bend_coef, alpha = 1.5, normal_coef = 1):
	"""
	Theta is coefficients of function
	Xs are source points
	Ys are target points
	B is bending energy matrix
	"""
	n,dim = Xs.shape
	m,_ = Exs.shape

	Y = np.r_[Ys, Eys]

	S = krig_kernel_mat(alpha, Xs, Epts, Exs)
	D = krig_mat_linear(Xs, Epts, Exs)
	B = bending_energynormal(S, D, dim)

	Q = np.c_[S, D]
	#WQ = wt_n[:, None]*Q
	WQ = Q
	H = Q.T.dot(WQ)
	#H[:n+m, :n+m] += bend_coef*B
	H[:n+m, :n+m] += bend_coef*B
	f = -WQ.T.dot(Y)

	A = np.c_[D.T, np.zeros((dim+1, dim+1))]

	Theta = solve_eqp1(H, f, A)

	targ = np.r_[Ys, Eys, np.zeros((dim+1, dim))]

	K = np.r_[np.c_[S, D], np.c_[D.T, np.zeros((dim+1, dim+1))]]
	
	point_diff = K.dot(Theta) - targ
	point_cost = np.sum(np.sum(np.abs(point_diff)**2, axis=-1))
	ba = Theta[:n+m]
	bend_cost = bend_coef*np.trace(ba.T.dot(B).dot(ba))

	return point_cost + bend_cost


def plot_krig_objective(bend_coefs):
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
		B = bending_energynormal(S, D, dim)

		Q = np.c_[S, D]
		#WQ = wt_n[:, None]*Q
		WQ = wt_n[:,None]*Q
		H = Q.T.dot(WQ)
		H[:n+m, :n+m] += bend_coef*B
		f = -WQ.T.dot(Y)

		A = np.c_[D.T, np.zeros((dim+1, dim+1))]

		Theta = solve_eqp1(H, f, A)

		print np.abs(np.sum(K.dot(Theta) - np.r_[Ys, Eys, np.zeros((dim+1,dim))]))

		Ys = K.dot(Theta)[:n, :]
		Eys = K.dot(Theta)[n:n+m,:]

		
	
	Y = np.r_[Ys, Eys]

	S = krig_kernel_mat(alpha, Xs, Epts, Exs)
	D1 = np.c_[np.ones((n,1)), Xs]
	D2 = np.c_[np.zeros((m,1)), np.ones((m,dim))]
	D = np.r_[D1, D2]
	B = bending_energynormal(S, D, dim)

	Q = np.c_[S, D]
	#WQ = wt_n[:, None]*Q
	WQ = wt_n[:,None]*Q
	H = Q.T.dot(WQ)
	H[:n+m, :n+m] += bend_coef*B
	f = -WQ.T.dot(Y)

	A = np.c_[D.T, np.zeros((dim+1, dim+1))]

	Theta = solve_eqp1(H, f, A)
	return K.dot(Theta) - np.r_[Ys, Eys, np.zeros((dim+1,dim))]

def tps_objective(Xs, Ys, bend_coef):
	n,d = Xs.shape

	K_nn = tps_kernel_matrix(Xs)
	Q = np.c_[np.ones((n,1)), Xs, K_nn]
	WQ = Q
	QWQ = Q.T.dot(WQ)
	H = QWQ
	H[d+1:,d+1:] += bend_coef * K_nn
    
	f = -WQ.T.dot(Ys)
    
	A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), Xs]].T
    
	Theta = solve_eqp1(H,f,A)

	targ = Ys

	point_diff = Q.dot(Theta) - targ
	point_cost = np.sum(np.sum(np.abs(point_diff)**2, axis=-1)**.5)
	ba = Theta[:n]
	bend_cost = bend_coef*np.trace(ba.T.dot(K_nn).dot(ba))

	return (point_cost + bend_cost), Theta

def plot_tps_objective(bend_coefs):
	costs = []
	Thetas = []
	Theta = np.zeros((n+d+1, d))
	for bend_coef in bend_coefs:
		cost, Theta1 = tps_objective(Xs, Ys, bend_coef)
		costs.append(cost)
		Thetas.append(Theta1 - Theta)
		Theta = Theta1
	return costs, Thetas

#derivative testing

def derivativex(f, x, epsilon = 1e-6):
	xpert = x.copy()
	xpert[:,0] = xpert[:,0] + epsilon
	return (f(xpert) - f(x))/epsilon

def derivativey(f, x, epsilon = 1e-6):
	xpert = x.copy()
	xpert[:,1] = xpert[:,1] + epsilon
	return (f(xpert)-f(x))/epsilon

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

def derivativexx(f, x, epsilon = 1e-6):
	xpert = x.copy()
	xpert[:,1] = xpert[:,1] + epsilon
	return (derivativey(f, xpert, epsilon) - derivativey(f,x,epsilon))/epsilon




"""
Derivative testing

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



"""















