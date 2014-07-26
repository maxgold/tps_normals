from tn_eval import krig_utils as ku
import numpy as np

def krig_eval(alpha, Xs, Epts, Exs, Ys, w_ng, lin_ag):
	"""
	Evaluates kriging function
	"""
	n, d = Ys.shape
	S = ku.krig_kernel_mat2(alpha, Xs, Epts, Exs, Exs, Ys, Ys)
	D = np.c_[np.ones((n,1)), Ys]
	
	return S[:n].dot(w_ng) + D.dot(lin_ag)

def krig_eval_landmark(alpha, Xs, Ys, w_ng, lin_ag):
	assert Xs.shape[1] == Ys.shape[1]
	n, d = Xs.shape
	m, _= Ys.shape
	S = ku.krig_mat_landmark2(alpha, Xs, Ys)
	D = np.c_[np.ones((m,1)), Ys]

	return S.dot(w_ng) + D.dot(lin_ag)

def krig_eval2(alpha, Xs, Epts, Exs, Ys, w_ng, lin_ag):
	"""
	Evaluates kriging function
	"""
	n, d = Ys.shape
	S = ku.krig_kernel_mat2(alpha, Xs, Epts, Exs, Ys)
	D = ku.krig_mat_linear2(Ys, Epts)
	
	return S[:n].dot(w_ng) + D[:n].dot(lin_ag)

def transform_normals1(alpha, Xs, Epts, Exs, Eypts, Eys, w_ng, lin_ag):
	"""
	transforms the normlas of the kriging function
	can only evaluate at Epts for now
	"""
	n, d = Eypts.shape
	S = ku.krig_kernel_mat2(alpha, Xs, Epts, Exs, Eys, Xs, Eypts)
	D = ku.krig_mat_linear(Xs, Eypts, Eys)

	return S[n:].dot(w_ng) + D[n:].dot(lin_ag)

def transform_normals2(alpha, Xs, Epts, Exs, Ys, w_ng, lin_ag):
	"""
	transforms the normlas of the kriging function
	can only evaluate at Epts for now
	"""
	n, d = Ys.shape
	S = ku.krig_kernel_mat2(alpha, Xs, Epts, Exs, Ys)
	D = ku.krig_mat_linear2(Ys, Epts, d)

	return S[n:].dot(w_ng) + D[n:].dot(lin_ag)
