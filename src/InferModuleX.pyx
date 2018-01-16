import numpy as np

cdef extern from "InferFunctions.h":
	void init_class(int _task_num, int _worker_num, int _class_num, int _true_label_num, int *_y, double *_alpha, double *_beta, double *_yDist, double *_cmat)
	void infer(int *pLabelMat, int *true_label)


def init_classX(task_num, worker_num, class_num, true_label_num, y, alpha, beta, yDist, cmat):
	cdef int _task_num = task_num
	cdef int _worker_num = worker_num
	cdef int _class_num = class_num
	cdef int _true_label_num = true_label_num
	cdef int[:] _y = y
	cdef double[:,:,:] _alpha = alpha
	cdef double[:] _beta = beta
	cdef double[:,:] _yDist = yDist
	cdef double[:,:,:] _cmat = cmat
	init_class(_task_num, _worker_num, _class_num, _true_label_num, &_y[0], &_alpha[0,0,0], &_beta[0], &_yDist[0,0], &_cmat[0,0,0])

def inferX(label_mat, true_label):
	cdef int[:,:] _label_mat = label_mat
	cdef int[:] _true_label = true_label
	infer(&_label_mat[0,0], &_true_label[0])