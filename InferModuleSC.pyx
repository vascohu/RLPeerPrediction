import numpy as np

cdef extern from "InferFunctionsSC.h":
	void init_class(int _task_num, int _worker_num, double *_yDist, double *_pVec)
	double infer(int *pLabelMat)

def init_classX(task_num, worker_num, yDist, pVec):
	cdef int _task_num = task_num
	cdef int _worker_num = worker_num
	cdef double[:,:] _yDist = yDist
	cdef double[:] _pVec = pVec
	init_class(_task_num, _worker_num, &_yDist[0,0], &_pVec[0])

def inferX(label_mat):
	cdef int[:,:] _label_mat = label_mat
	infer(&_label_mat[0,0])