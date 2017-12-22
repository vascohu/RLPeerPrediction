import numpy as np

cdef extern from "InferFunctionsSC.h":
<<<<<<< HEAD
	void init_class(int _task_num, int _worker_num, double *_yDist, double *_pVec)
	double infer(int *pLabelMat)

def init_classX(task_num, worker_num, yDist, pVec):
=======
	void init_class(int _task_num, int _worker_num, double *_yDist, double *_pVec, double *_xVec)
	double infer(int *pLabelMat)
	void deconstruct()

def init_classX(task_num, worker_num, yDist, pVec, xVec):
>>>>>>> bafee7413e37b17cabdd70d687eef73a6b889cca
	cdef int _task_num = task_num
	cdef int _worker_num = worker_num
	cdef double[:,:] _yDist = yDist
	cdef double[:] _pVec = pVec
<<<<<<< HEAD
	init_class(_task_num, _worker_num, &_yDist[0,0], &_pVec[0])

def inferX(label_mat):
	cdef int[:,:] _label_mat = label_mat
	infer(&_label_mat[0,0])
=======
	cdef double[:] _xVec = xVec
	init_class(_task_num, _worker_num, &_yDist[0,0], &_pVec[0], &_xVec[0])

def inferX(label_mat):
	cdef int[:,:] _label_mat = label_mat
	return infer(&_label_mat[0,0])

def deconstructX():
	deconstruct()
>>>>>>> bafee7413e37b17cabdd70d687eef73a6b889cca
