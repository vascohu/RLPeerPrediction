# CrowdModule describes the crowdsourcing market, including tasks and workers
import abc
import numpy as np
import MechModule


# How to describe a task?
# Here, we use two parameters: the true label and the difficulty level
class Task:
    # True label (should be '1' or '2')
    true_label = 0

    # Difficulty level (should be a value in [0,1]
    difficulty = 0

    def __init__(self):
        """Uniformly generate the true labels and difficulty level"""
        self.true_label = np.random.randint(1, 3)
        self.difficulty = np.random.rand()


# How to describe a worker?
# Firstly, we should announce the incentive mechanism to workers. Thus, an incentive mechanism mirror is needed.
# Secondly, we provide a task to a worker and get the label. Thus, a labeling function is needed.
# The base class of worker
class Worker(object):
    __metaclass__ = abc.ABCMeta

    # The mirror of the incentive mechanism
    mirror_of_mech = None

    @abc.abstractmethod
    def labeling(self, task: Task):
        """Decide the label for a task"""
        return


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
        return 0.5*self.mirror_of_mech.B - task.difficulty

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

    def __init__(self, task_number: int, worker_number: int, mech: MechModule.MechBase):
        # Set the numbers of tasks and workers
        self.n_task = task_number
        self.n_worker = worker_number
        self.worker_num_per_task = self.n_worker - 1
        # Generate workers
        for j in range(worker_number):
            new_worker = QRWorker(mech)
            self.worker_list.append(new_worker)
        # Generate the assign table
        self.task_assign_table = self.assign_task()

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
            new_task = Task()
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

