# RLModule is responsible to decide the action
import abc
import numpy as np


# The base class of RL
class RLBase(object):
    __metaclass__ = abc.ABCMeta

    # The discount factor
    gamma = 0.8

    # The action set (prices goes from the smallest to the largest)
    ActionSet = [1, 2]

    @abc.abstractmethod
    def decide(self):
        """Decide the action a_t"""
        return

    @abc.abstractmethod
    def observe(self, a, r, s):
        """Observe <reward_t, state_t+1> """
        return


# Gaussian Process SARSA
class GpSarsa(RLBase):

    def __init__(self):
        # The noisy level of the Gaussian process
        self.sigma = 0.1
        # Observation history
        self.Hist = []  # <State, Action>
        self.rHist = []  # Reward
        # The covariance matrix
        self.Cov = []
        # Current State
        self.S = []
        # H matrix
        self.invH = []
        # Gaussian Process Parameters
        self.Alpha = None
        self.C = None

    def decide(self):
        """We use Thompson sampling to decide the next-step action."""
        if len(self.Hist) == 0:
            '''At the first step, choose the lowest price'''
            return RLBase.ActionSet[0]
        else:
            '''Firstly, update the Gaussian process model'''
            self.gpRegression()
            '''Secondly, compute the prediction of the Gaussian process'''
            q = []
            z = self.S.copy().append(self.ActionSet[0])
            for a in RLBase.ActionSet:
                z[-1] = a
                q.append(self.gpPredict(z))
            '''Thirdly, select the action with the largest Q value'''
            pos = q.index(max(q))
            return RLBase.ActionSet[pos]

    def observe(self, a, r, s):
        """Observe the environment change after action a"""
        '''Add the data to the observation history'''
        if len(self.S)>0:
            self.Hist.append(self.S+[a])
            self.rHist.append(r)
        '''Update the current state'''
        self.S = s.copy()

    # noinspection PyUnresolvedReferences
    @classmethod
    def kernel(cls, z1: np.array, z2: np.array) -> float:
        diff = (z1 - z2)**2
        '''The difference between different workers'''
        distW = np.exp(-1.0*(diff[0:-1:2]+diff[1:-1:2]))
        '''The difference between actions'''
        distA = np.exp(-0.01*diff[-1])
        # noinspection PyTypeChecker
        return np.sum(distW)*distA
        #return np.exp(-0.5*np.sum(np.abs(diff)))

    def gpRegression(self):
        """Compute the Gaussian Process model"""
        ''''Compute the covariance between new observation and old ones'''
        k_cov = [self.kernel(np.asarray(self.Hist[-1]), np.asarray(z)) for z in self.Hist]
        '''Add these data to the covariance table'''
        for (c1, c2) in zip(self.Cov, k_cov):
            c1.append(c2)
        self.Cov.append(k_cov)
        '''Compute the inverse of the covariance'''
        T = len(self.Cov)
        matCov = np.asarray(self.Cov)
        invCov = np.linalg.inv(matCov+self.sigma*self.sigma*np.identity(T))
        '''Update the inverse H matrix'''
        for i in range(T-1):
            self.invH[i].append(RLBase.gamma**(T-i-1))
        self.invH.append([0]*(T-1)+[1])
        matInvH = np.asarray(self.invH)
        '''Compute alpha and C matrix'''
        self.Alpha = np.dot(np.dot(invCov, matInvH), self.rHist)
        self.C = invCov
        return

    def gpPredict(self, z):
        """Use the Gaussian Process model to predict Q(z=<s,a>)"""
        '''Compute the mean value and variance'''
        k_cov = [self.kernel(np.asarray(z), np.asarray(el)) for el in self.Hist]
        vec_k_cov = np.asarray(k_cov)
        k0 = self.kernel(np.asarray(z), np.asarray(z))
        meanVal = vec_k_cov.dot(self.Alpha)
        varVal = k0 - vec_k_cov.dot(np.dot(self.C,vec_k_cov))
        '''Generate a sample from the prediction'''
        return np.random.normal(meanVal, np.sqrt(varVal))