# CrowdModule describes the crowdsourcing market, including tasks and workers
import abc
import numpy as np
import MechModule


# How to describe a task?
# Here, we use two parameters: the true label and the difficulty level
class Task:
    # Task id
    id = 0

    # True label (should be '1' or '2')
    true_label = 0

    # Difficulty level (should be a value in [0,1]
    difficulty = 0

    # The generating distribution
    p = 0.5

    def __init__(self, task_id):
        """Uniformly generate the true labels and difficulty level"""
        self.id = task_id
        #self.true_label = np.random.randint(1, 3)
        th = np.random.rand()
        if th<Task.p:
            self.true_label = 1
        else:
            self.true_label = 2
        self.difficulty = np.random.rand()


# How to describe a worker?
# Firstly, we should announce the incentive mechanism to workers. Thus, an incentive mechanism mirror is needed.
# Secondly, we provide a task to a worker and get the label. Thus, a labeling function is needed.
# The base class of worker
class Worker(object):
    __metaclass__ = abc.ABCMeta

    # The mirror of the incentive mechanism
    mirror_of_mech = None
    high_effort_cost = 0.02

    @abc.abstractmethod
    def labeling(self, task: Task):
        """Decide the label for a task"""
        return

    @abc.abstractmethod
    def evolve(self, rewards: list):
        """Adapt their strategies according to the rewards"""
        return

class RWorker(Worker):
    def __init__(self, mech: MechModule.MechBase):
        # Set the mechanism mirror
        self.mirror_of_mech = mech

    def labeling(self, task: Task):
            th = np.random.rand()
            if th < 0.90:
                return task.true_label
            else:
                return (3 - task.true_label)

    def evolve(self, rewards: list):
            pass


# The quantal response Worker
class QRWorker(Worker):

    # The QR parameter
    lam = 0.0

    def __init__(self, mech: MechModule.MechBase):
        # Set the mechanism mirror
        self.mirror_of_mech = mech
        # Set the QR parameter
        self.lam = 4.0

    def high_effort_value(self, task: Task):
        """Calculate the high-effort value of a task"""
        '''We regard the cost of high-effort equals the difficult level'''
        return 0.5*self.mirror_of_mech.B - task.difficulty - Worker.high_effort_cost

    def low_effort_value(self, task: Task):
        """Calculate the low-effort cost of a task"""
        return 0.0

    def strategy(self, task: Task):
        """Decide the reporting strategy"""
        '''The value of a strategy'''
        value = np.zeros(2)
        value[0] = self.high_effort_value(task)
        value[1] = self.low_effort_value(task)
        '''The QR probability'''
        prob = np.exp(self.lam * value)
        prob /= np.sum(prob)
        '''Decide the strategy'''
        return np.random.choice(np.arange(0, 2), p = prob)

    def labeling(self, task: Task):
        if self.strategy(task) == 0:
            return task.true_label
        else:
            return np.random.choice([1, 2])

    def evolve(self, rewards: list):
        pass

# The MWUA Worker
class MWUA_Worker(Worker):

    def __init__(self, mech: MechModule.MechBase):
        # Set the mechanism mirror
        self.mirror_of_mech = mech
        # Set the MWUA parameter
        self.epsilon = 0.1
        # The labeled task id
        self.task_ids = []
        # The corresponding strategies
        self.s_list = []
        # The labeling strategy
        self.strategy = np.ones(2)*0.5
        self.strategy[0] = 0.2
        self.strategy[1] = 0.8

    def labeling(self, task: Task):
        self.task_ids.append(task.id)
        s = np.random.choice(np.arange(self.strategy.shape[0]), p=self.strategy)
        self.s_list.append(s)
        if s==0:
            return task.true_label
        else:
            return np.random.choice([1, 2])

    def evolve(self, rewards: list):
        u = [[] for _ in range(self.strategy.shape[0])]
        for id in self.task_ids:
            i = self.task_ids.index(id)
            s = self.s_list[i]
            u[s].append(rewards[id])
        uu = np.zeros(self.strategy.shape[0])
        for j in range(self.strategy.shape[0]):
            if len(u[j]) == 0:
                uu[j] = 0
            else:
                if j==0:
                    uu[j] = np.mean(u[j]) - Worker.high_effort_cost
                else:
                    uu[j] = np.mean(u[j])
        uu -= np.min(uu)
        for j in range(self.strategy.shape[0]):
            self.strategy[j] *= 1 + uu[j]*self.epsilon
        self.strategy /= np.sum(self.strategy)
        self.task_ids.clear()
        self.s_list.clear()

class TRWorker(Worker):
    P_H = 0.8
    P_L = 0.5

    def __init__(self, mech: MechModule.MechBase, id: int):
        # Set the mechanism mirror
        self.mirror_of_mech = mech
        # High or Low
        self.e = 1
        # True or False
        self.r = 1
        # The probability
        self.P_H = TRWorker.P_H #np.random.rand()*0.5+0.5#0.8#TRWorker.P_H
        self.P_L = TRWorker.P_L
        # Set the first worker
        if id==0:
            self.e = 1
            self.r = 1
            self.P_H = TRWorker.P_H

    def labeling(self, task: Task):
        th = np.random.rand()

        if self.e==1:
            p = self.P_H
        else:
            p = self.P_L

        if self.r == 1:
            if th < p:
                return task.true_label
            else:
                return (3 - task.true_label)
        else:
            if th < p:
                return (3-task.true_label)
            else:
                return task.true_label

    def evolve(self, rewards: list):
        pass

# Crowdsourcing Market
class CrowdMarket:

    # Task Number
    n_task = 0

    # Worker Number
    n_worker = 0

    # Worker per task
    worker_num_per_task = 0

    # Task_list
    task_list = []

    # Worker_list
    worker_list = []

    # Task Assign Table
    task_assign_table = None

    # The mechanism
    mech = None

    # worker type
    workerType = None

    def __init__(self, task_number: int, worker_number: int, mech: MechModule.MechBase, workerType: str):
        # Set the numbers of tasks and workers
        self.n_task = task_number
        self.n_worker = worker_number
        if isinstance(mech, MechModule.DG13):
            self.worker_num_per_task = self.n_worker - 1
        else:
            self.worker_num_per_task = self.n_worker
        # Generate workers
        self.worker_init()
        # Generate the assign table
        self.task_assign_table = self.assign_task()
        self.mech = mech
        self.workerType = workerType

    def worker_init(self):
        self.worker_list.clear()
        for j in range(self.n_worker):
            if self.workerType == "rational":
                new_worker = RWorker(self.mech)
            elif self.workerType == "QR":
                new_worker = QRWorker(self.mech)
            elif self.workerType == "MWUA":
                new_worker = MWUA_Worker(self.mech)
            else:
                if j%3 == 0:
                    new_worker = RWorker(self.mech)
                elif j%3 == 1:
                    new_worker = QRWorker(self.mech)
                else:
                    new_worker = MWUA_Worker(self.mech)
            self.worker_list.append(new_worker)

    def assign_task(self):
        """Assign tasks to workers"""
        task_assign_table = [[] for _ in range(self.n_worker)]
        worker_id = 0
        for i in range(self.n_task):
            for k in range(self.worker_num_per_task):
                # print(worker_id)
                task_assign_table[worker_id].append(i)
                worker_id += 1
                if worker_id >= self.n_worker:
                    worker_id = 0
        return task_assign_table

    def get_label_mat(self, n_true_label: int):
        """Generate the label matrix"""
        '''Generate the tasks'''
        self.task_list.clear()
        for i in range(self.n_task):
            new_task = Task(i)
            self.task_list.append(new_task)
        '''Generate the labels'''
        label_mat = np.zeros(shape=(self.n_task, self.n_worker + 1), dtype=int)
        for j in range(self.n_worker):
            worker = self.worker_list[j]
            for i in self.task_assign_table[j]:
                task = self.task_list[i]
                label_mat[i, j] = worker.labeling(task)
        '''Add the true labels'''
        for i in range(n_true_label):
            task = self.task_list[i]
            label_mat[i, -1] = task.true_label
        return label_mat

    def get_label_mat_NTL(self):
        """Generate the label matrix"""
        '''Generate the tasks'''
        self.task_list.clear()
        for i in range(self.n_task):
            new_task = Task(i)
            self.task_list.append(new_task)
        '''Generate the labels'''
        label_mat = np.zeros(shape=(self.n_task, self.n_worker), dtype=int)
        for j in range(self.n_worker):
            worker = self.worker_list[j]
            for i in self.task_assign_table[j]:
                task = self.task_list[i]
                label_mat[i, j] = worker.labeling(task)
        return label_mat

    def get_true_label(self):
        """Return the true label list"""
        true_label = []
        for task in self.task_list:
            true_label.append(task.true_label)
        return true_label

    def get_task_difficulty(self):
        """Return the task difficulty"""
        difficulty = []
        for task in self.task_list:
            difficulty.append(task.difficulty)
        return difficulty

    def evolve(self, reward_mat: list):
        # Workers evolve as the obtained rewards
        for (worker, rewards) in zip(self.worker_list, reward_mat):
            worker.evolve(rewards)