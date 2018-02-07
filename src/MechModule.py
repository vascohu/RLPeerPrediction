# MechModule receives the parameter from RLModule and decides the payment for agents' actions

import abc
import numpy as np


# The base class of incentive mechanisms
class MechBase(object):
    __metaclass__ = abc.ABCMeta

    # The incentive level
    B = 1.0

    @abc.abstractmethod
    def set(self, parameters: list):
        """Set the mechanism parameters"""
        """We use a list to represent the parameters to incorporate multiple parameters in future versions"""
        return

    @abc.abstractmethod
    def pay(self, label_mat: np.matrix, p_vec = None):
        """Decide the payment for workers """
        return


# DG13 Mechanism
class DG13(MechBase):

    # The parameter of DG13 mechanism
    d = 10

    def __init__(self):
        # The label mat
        self.label_mat = None

    def set(self, parameters: list):
        self.B = parameters[0]

    def pay(self, label_mat: np.matrix, p_vec = None):
        """Compute the total reward and the detailed rewards for each task"""
        p = 0
        reward_mat = []
        self.label_mat = label_mat
        for i in range(label_mat.shape[1] - 1):
            r = self.reward_a_agent(i)
            p += np.sum(r)
            reward_mat.append(r)
        return (p, reward_mat)

    def reward_a_agent(self, agent_id: int):
        """Compute the reward for one agent"""
        reward_vec = []
        for i in range(self.label_mat.shape[0]):
            reward = 0
            if self.label_mat[i, agent_id] != 0:
                ref_agent_id = self.ref_agent(agent_id, i)
                if self.label_mat[i, agent_id] == self.label_mat[i, ref_agent_id]:
                    reward += 1
                reward -= self.non_overlapping_set(i, agent_id, ref_agent_id)
            reward_vec.append(reward*self.B)
        return reward_vec

    def ref_agent(self, agent_id: int, task_id: int):
        """Randomly select the reference agent"""
        if self.label_mat[task_id, -1] != 0:
            return self.label_mat.shape[1] - 1
        else:
            id_set = []
            for w in range(self.label_mat.shape[1]):
                if w != agent_id and self.label_mat[task_id, w] != 0:
                    id_set.append(w)
            return np.random.choice(id_set)

    def non_overlapping_set(self, task_id: int, agent_id1: int, agent_id2: int):
        """Compute the reports on the non-overlapping set"""
        '''Calculate the non-overlapping task sets'''
        non_overlapping_set1 = []
        non_overlapping_set2 = []
        for i in range(self.label_mat.shape[0]):
            if i != task_id:
                if (self.label_mat[i, agent_id1] != 0) and (self.label_mat[i, agent_id2] == 0):
                    non_overlapping_set1.append(i)
                elif (self.label_mat[i, agent_id1] == 0) and (self.label_mat[i, agent_id2] != 0):
                    non_overlapping_set2.append(i)
        if len(non_overlapping_set1) < self.d or len(non_overlapping_set2) < self.d:
            print("The inappropriate d!!!")
            exit()
        '''Sample d non-overlapping tasks'''
        set1 = np.random.choice(non_overlapping_set1, self.d, replace=False)
        set2 = np.random.choice(non_overlapping_set2, self.d, replace=False)
        '''Compute the penalty item in DG13'''
        p1 = self.count_a_set(agent_id1, set1)
        p2 = self.count_a_set(agent_id2, set2)
        penalty = (p1[0]*p2[0]+p1[1]*p2[1])/self.d/self.d
        return penalty

    def count_a_set(self, worker_id: int, task_set: list):
        """Count the number of '1' and '2' for the task set"""
        c = [0]*2
        for i in task_set:
            label = self.label_mat[i, worker_id]
            c[label-1] += 1
        return c


class BayesMech(object):

    def set(self, parameters: list):
        self.B = parameters[0]

    def pay(self, label_mat: np.matrix, p_vec = None):
        """Compute the total reward and the detailed rewards for each task"""
        reward_mat = []
        p = 0
        for i in range(label_mat.shape[1]):
            reward_vec = []
            for j in range(label_mat.shape[0]):
                if label_mat[j, i] != 0:
                    reward_vec.append((p_vec[j,i]-0.5)*self.B)
            p += np.sum(reward_vec)
            reward_mat.append(reward_vec)
        return (p, reward_mat)


