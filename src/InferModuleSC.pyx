import numpy as np

cdef extern from "InferFunctionsSC.h":
	void init_class(long _task_num, long _worker_num, double *_yDist, double *_pVec, double *_xVec, double *_pDist)
	double infer(long *pLabelMat)
	void deconstruct()

def init_classX(task_num, worker_num, yDist, pVec, xVec, p_dist):
	cdef long _task_num = task_num
	cdef long _worker_num = worker_num
	cdef double[:,:] _yDist = yDist
	cdef double[:] _pVec = pVec
	cdef double[:] _xVec = xVec
	cdef double[:,:] _pDist = p_dist
	init_class(_task_num, _worker_num, &_yDist[0,0], &_pVec[0], &_xVec[0], &_pDist[0,0])

def inferX(label_mat):
	cdef long[:,:] _label_mat = label_mat
	return infer(&_label_mat[0,0])

def deconstructX():
	deconstruct()