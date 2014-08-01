from tn_rapprentice import krig_utils as ku
import numpy as np

def krig_eval(alpha, Xs, Epts, Exs, Ys, w_ng, lin_ag, trans_g):
	"""
	Evaluates kriging function
	"""
	n, d = Ys.shape
	S = ku.krig_kernel_mat2(alpha, Xs, Epts, Exs, Exs, Ys, Epts)
	
	return S[:n].dot(w_ng) + Ys.dot(lin_ag) + trans_g[None,:]

def krig_eval_landmark(alpha, Xs, Ys, w_ng, lin_ag):
	assert Xs.shape[1] == Ys.shape[1]
	n, d = Xs.shape
	m, _= Ys.shape
	S = ku.krig_mat_landmark2(alpha, Xs, Ys)
	D = np.c_[np.ones((m,1)), Ys]

	return S.dot(w_ng) + D.dot(lin_ag)

def transform_normals1(alpha, Xs, Epts, Exs, Eypts, Eys, w_ng, lin_ag):
	"""
	transforms the normlas of the kriging function
	can only evaluate at Epts for now
	"""
	n, d = Eypts.shape

	S = ku.krig_kernel_mat2(alpha, Xs, Epts, Exs, Eys, Xs, Eypts)	
	D = ku.krig_mat_linear(Xs, Eypts, Eys)

	return S[n:].dot(w_ng) + D[n:, 1:].dot(lin_ag)


