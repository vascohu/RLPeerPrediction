# Inference module is responsible to infer the state of workers and calculate rewards
import abc
import numpy as np
import InferModuleSC
#from scipy.special import gammaln
#import time

def random_sample(start: int, p: np.array):
    r = np.random.rand()
    i = 0
    for pr in range(p.shape[0]):
        if r < pr:
            break
        else:
            r -= p[i]
            i += 1
    return (i+start)

# The base class of inference
class InferBase(object):
    __metaclass__ = abc.ABCMeta

    # Workers states
    belief = None

    # The expected accuracy
    ex_accuracy = 0.0

    # The expected log-prob-ratio
    ex_x = 0.0

    # The reward brought
    R = 0.0

    # The discount of reward
    eta = 0.0001

    @abc.abstractmethod
    def infer(self, label_mat: np.matrix, true_label: list = None):
        """Infer the states and expected accuracy"""
        return

    def reward(self, payment: float):
        return (self.ex_accuracy)**10 - self.eta*payment

    def real_reward(self, accuracy: float, payment: float):
        return accuracy**10 - self.eta*payment

    @abc.abstractmethod
    def test(self, label_mat: np.matrix, true_label: list):
        """Test the inference module"""
        return


'''class GibbsSampling(InferBase):

    def __init__(self, _task_num: int, _worker_num: int, _class_num: int, _true_label_num: int = 0):
        self.task_num = _task_num
        self.worker_num = _worker_num
        self.class_num = _class_num
        self.true_label_num = _true_label_num
        self.sample = np.zeros(shape=self.task_num, dtype=int)
        self.alpha = np.ones(shape=(self.worker_num, self.class_num, self.class_num))
        self.beta = np.ones(shape=self.class_num)
        self.y_dist = np.zeros(shape=(self.task_num-self.true_label_num, self.class_num))
        self.b = np.zeros(shape=self.alpha.shape)
        self.belief = np.zeros(2 * self.worker_num)
        InferModuleX.init_classX(self.task_num, self.worker_num, self.class_num, self.true_label_num,
                                self.sample, self.alpha, self.beta, self.y_dist, self.b)

    def infer(self, label_mat: np.matrix, true_label: list = None):
        InferModuleX.infer(label_mat, np.asarray(true_label))

    def test(self, label_mat: np.matrix, true_label: list):
        InferModuleX.inferX(label_mat, np.asarray(true_label))
        '''Calculate the belief about workers'''
        for j in range(self.worker_num):
            self.belief[2*j] = self.b[j, 0, 0]/np.sum(self.b[j, 0, :])
            self.belief[2*j+1] = self.b[j, 1, 1]/np.sum(self.b[j, 1, :])
        '''Calculate the expected accuracy'''
        self.ex_accuracy = 0
        accuracy = 0
        for i in range(self.task_num-self.true_label_num):
            self.ex_accuracy += np.max(self.y_dist[i, :])
            label = np.argmax(self.y_dist[i, :])
            if label == true_label[i+self.true_label_num]-1:
                accuracy += 1
        self.ex_accuracy /= (self.task_num-self.true_label_num)
        accuracy /= (self.task_num-self.true_label_num)
        return accuracy
'''

class GibbsSamplingSC(InferBase):
    def __init__(self, _task_num: int, _worker_num: int):
        self.task_num = _task_num
        self.worker_num = _worker_num
        self.y_dist = np.zeros(shape=(_task_num, 2))
        self.p_vec = np.zeros(shape=(_worker_num))
        self.x_vec = np.zeros(shape=(_task_num))
        InferModuleSC.init_classX(self.task_num, self.worker_num, self.y_dist, self.p_vec, self.x_vec)

    def __del__(self):
        pass
        #InferModuleSC.deconstructX()

    def infer(self, label_mat: np.matrix, true_label: list=None):
        self.ex_accuracy = InferModuleSC.inferX(label_mat)
        self.ex_prob_ratio()

    def test(self, label_mat: np.matrix, true_label: list):
        self.ex_accuracy = InferModuleSC.inferX(label_mat)
        self.ex_prob_ratio()
        accuracy = 0
        for i in range(self.task_num):
            #label = np.argmax(self.y_dist[i, :])
            if self.x_vec[i]>=0:
                label = 0
            else:
                label = 1
            if label == true_label[i]-1:
                accuracy += 1
        accuracy /= self.task_num
        return accuracy

    def ex_prob_ratio(self):
        abs_log_ratio = np.absolute(self.x_vec)
        temp = np.mean(abs_log_ratio)
        self.ex_x = np.exp(temp)/(1+np.exp(temp))