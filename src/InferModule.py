# Inference module is responsible to infer the state of workers and calculate rewards
import abc
import numpy as np
#import InferModuleX
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


# class GibbsSampling(InferBase):
#
#     def __init__(self, _task_num: int, _worker_num: int, _class_num: int, _true_label_num: int = 0):
#         self.task_num = _task_num
#         self.worker_num = _worker_num
#         self.class_num = _class_num
#         self.true_label_num = _true_label_num
#         self.sample = np.zeros(shape=self.task_num, dtype=int)
#         self.alpha = np.ones(shape=(self.worker_num, self.class_num, self.class_num))
#         self.beta = np.ones(shape=self.class_num)
#         self.y_dist = np.zeros(shape=(self.task_num-self.true_label_num, self.class_num))
#         self.b = np.zeros(shape=self.alpha.shape)
#         self.belief = np.zeros(2 * self.worker_num)
#         self.p_dist = np.zeros(self.task_num, self.worker_num)
#         InferModuleX.init_classX(self.task_num, self.worker_num, self.class_num, self.true_label_num,
#                                 self.sample, self.alpha, self.beta, self.y_dist, self.b, self.p_dist)
#
#     def infer(self, label_mat: np.matrix, true_label: list = None):
#         InferModuleX.infer(label_mat, np.asarray(true_label))
#
#     def test(self, label_mat: np.matrix, true_label: list):
#         InferModuleX.inferX(label_mat, np.asarray(true_label))
#         '''Calculate the belief about workers'''
#         for j in range(self.worker_num):
#             self.belief[2*j] = self.b[j, 0, 0]/np.sum(self.b[j, 0, :])
#             self.belief[2*j+1] = self.b[j, 1, 1]/np.sum(self.b[j, 1, :])
#         '''Calculate the expected accuracy'''
#         self.ex_accuracy = 0
#         accuracy = 0
#         for i in range(self.task_num-self.true_label_num):
#             self.ex_accuracy += np.max(self.y_dist[i, :])
#             label = np.argmax(self.y_dist[i, :])
#             if label == true_label[i+self.true_label_num]-1:
#                 accuracy += 1
#         self.ex_accuracy /= (self.task_num-self.true_label_num)
#         accuracy /= (self.task_num-self.true_label_num)
#         return accuracy


class GibbsSamplingSC(InferBase):
    def __init__(self, _task_num: int, _worker_num: int):
        self.task_num = _task_num
        self.worker_num = _worker_num
        self.y_dist = np.zeros(shape=(_task_num, 2))
        self.p_vec = np.zeros(shape=(_worker_num))
        self.x_vec = np.zeros(shape=(_task_num))
        self.p_dist = np.zeros(shape=(self.task_num, self.worker_num))
        InferModuleSC.init_classX(self.task_num, self.worker_num, self.y_dist, self.p_vec, self.x_vec, self.p_dist)

    def __del__(self):
        pass
        #InferModuleSC.deconstructX()

    def infer(self, label_mat: np.matrix, true_label: list=None):
        self.ex_accuracy = InferModuleSC.inferX(label_mat)
        self.ex_prob_ratio()

    def test(self, label_mat: np.matrix, true_label: list):
        self.ex_accuracy = InferModuleSC.inferX(label_mat)
        # print(self.y_dist)
        self.ex_prob_ratio()
        accuracy = 0
        for i in range(self.task_num):
            #label = np.argmax(self.y_dist[i, :])
            if self.x_vec[i]>=0:
            #if self.y_dist[i,0]> self.y_dist[i,1]:
                label = 0
            else:
                label = 1
            if label == true_label[i]-1:
                accuracy += 1
        accuracy /= self.task_num
        return accuracy

    def ex_prob_ratio(self):
        abs_log_ratio = np.absolute(self.x_vec)
        #temp = np.mean(abs_log_ratio)
        temp = np.exp(abs_log_ratio)/(1+np.exp(abs_log_ratio))
        #self.ex_accuracy = np.mean(np.max(self.y_dist, axis=1))
        #self.ex_x = np.exp(temp)/(1+np.exp(temp))
        self.ex_x = np.mean(temp)
        self.belief = self.p_vec
        self.ex_accuracy = self.ex_x



"""
    # Number of samples
    sample_num = 100

    # Number of burn-in samples
    burn_num = 100

    # The sampling interval
    interval = 5

    # The distribution of true labels
    y_dist = None

    def __init__(self, _task_num: int, _worker_num: int, _class_num: int, _true_label_num: int = 0):
        self.task_num = _task_num
        self.worker_num = _worker_num
        self.class_num = _class_num
        self.sample = np.zeros(shape=self.task_num, dtype=int)
        self.alpha = np.ones(shape=(self.worker_num, self.class_num, self.class_num))
        self.beta = np.ones(shape=self.class_num)
        self.true_label_num = _true_label_num
        self.y_dist = np.zeros(shape=(self.task_num-self.true_label_num, self.class_num))
        self.b = np.zeros(shape=self.alpha.shape)
        self.belief = np.zeros(2 * self.worker_num)

    def calc_prior_dist(self, label_mat: np.matrix = None, true_label: list = None):
        '''Calculate the prior distribution'''
        self.alpha.fill(1.0)
        self.beta.fill(1.0)
        if self.true_label_num == 0:
            '''If there are no true labels, use optimistic priors.'''
            for wm in self.alpha:
                for k in range(self.class_num):
                    wm[k, k] += 1
        else:
            '''If there are true labels, compute the Dir distribution.'''
            for i in range(self.true_label_num):
                k = true_label[i]-1
                for j in range(self.worker_num):
                    g = label_mat[i, j]-1
                    if g >= 0:
                        self.alpha[j, k, g] += 1
                self.beta[k] += 1

    def init_y_alpha_beta(self, label_mat: np.matrix):
        '''Calculate the initial values'''
        for i in np.arange(self.true_label_num, self.task_num):
            '''Majority voting to compute the probability'''
            votes = np.zeros(self.class_num)
            for label in np.ravel(label_mat[i, :]):
                votes[label-1] += 1
            p = votes/np.sum(votes)
            '''Generate the label and Initialize'''
            #label = np.random.choice(np.arange(0, self.class_num), p=p)
            label = random_sample(0, p)
            self.sample[i] = label
            self.beta[label] += 1
            for j in range(self.worker_num):
                g = label_mat[i, j] - 1
                if g >= 0:
                    self.alpha[j, label, g] += 1

    def update_alpha_beta_y(self, label_mat: np.matrix, i: int, yn: int):
        '''Update the alpha tensor, beta vector with new yn for task i'''
        y0 = self.sample[i]
        if y0 != yn:
            '''Update the beta vector'''
            self.beta[y0] -= 1
            self.beta[yn] += 1
            '''Update the alpha tensor'''
            for j in range(self.worker_num):
                g = label_mat[i, j] - 1
                if g >= 0:
                    self.alpha[j][y0][g] -= 1
                    self.alpha[j][yn][g] += 1
            '''Update y'''
            self.sample[i] = yn

    @classmethod
    def log_m_beta(cls, x: np.array) -> float:
        '''The Log Beta function'''
        log_prob = 0
        sum_x = 0
        for i in range(x.shape[0]):
            log_prob += gammaln(x[i])
            sum_x += x[i]
        log_prob -= gammaln(sum_x)
        return log_prob

    def generate_one_label(self, label_mat: np.matrix, i: int):
        '''Generate one label as the Gibbs sampling'''
        '''Calculate the conditional probability'''
        log_prob = np.zeros(self.class_num)
        for k in range(self.class_num):
            self.update_alpha_beta_y(label_mat, i, k)
            log_p = self.log_m_beta(self.beta)
            for wm in self.alpha:
                for g in range(self.class_num):
                    log_p += self.log_m_beta(wm[g, :])
            log_prob[k] = log_p
        log_prob -= np.max(log_prob)
        prob = np.exp(log_prob)
        prob /= np.sum(prob)
        '''Generate a new label'''
        label = random_sample(0, prob)
        # np.random.choice(np.arange(self.class_num), p=prob)
        self.update_alpha_beta_y(label_mat, i, label)

    def gs_burn_in(self, label_mat: np.matrix):
        '''The burn-in process in Gibbs sampling'''
        self.init_y_alpha_beta(label_mat)
        for t in range(self.burn_num):
            for i in np.arange(self.true_label_num, self.task_num):
                self.generate_one_label(label_mat, i)

    def infer(self, label_mat: np.matrix, true_label: list = None):
        '''Infer the states and expected accuracy'''
        '''Compute the prior distribution'''
        if true_label is None:
            self.calc_prior_dist()
        else:
            self.calc_prior_dist(label_mat, true_label)
        '''Burn In'''
        self.gs_burn_in(label_mat)
        '''Generate samples'''
        self.y_dist.fill(0.0)
        self.b.fill(0.0)
        for t in range(self.sample_num):
            '''Take one sample in every interval'''
            for n in range(self.interval):
                for i in np.arange(self.true_label_num, self.task_num):
                    self.generate_one_label(label_mat, i)
            '''Update the y_dist and self.b'''
            for i in np.arange(self.true_label_num, self.task_num):
                self.y_dist[i-self.true_label_num, self.sample[i]] += 1.0
                self.b += self.alpha
        self.y_dist /= self.sample_num
        self.b /= self.sample_num
        '''Calculate the expected accuracy'''
        self.ex_accuracy = 0
        for dist in self.y_dist:
            self.ex_accuracy += np.max(dist)
        self.ex_accuracy /= (self.task_num-self.true_label_num)

    def test(self, label_mat: np.matrix, true_label: list):
        '''Test the inference results and return the real accuracy'''
        '''Compute the prior distribution'''
        self.calc_prior_dist(label_mat, true_label)
        '''Burn In'''
        self.gs_burn_in(label_mat)
        '''Generate samples'''
        self.y_dist.fill(0.0)
        self.b.fill(0.0)
        self.belief.fill(0.0)
        for t in range(self.sample_num):
            start = time.time()
            '''Take one sample in every interval'''
            for n in range(self.interval):
                for i in np.arange(self.true_label_num, self.task_num):
                    self.generate_one_label(label_mat, i)
            '''Update the y_dist and self.b'''
            for i in np.arange(self.true_label_num, self.task_num):
                self.y_dist[i-self.true_label_num, self.sample[i]] += 1.0
            self.b += self.alpha
            end = time.time()
            print(start - end)
        self.y_dist /= self.sample_num
        self.b /= self.sample_num
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
"""
