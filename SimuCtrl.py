import numpy as np
import RLModule
import InferModule
import MechModule

worker_num = 4
task_num = 100

'''
for i in range(20):
    alpha = np.zeros(shape=(worker_num, 2, 2))
    for j in range(worker_num):
        for k in range(2):
            r = np.random.beta(3,1)
            alpha[j, k, k] = r
            alpha[j, k, 1-k] = 1.0 -r

    true_label =[]
    label = np.zeros(shape=(task_num, worker_num), dtype=int)
    for i in range(task_num):
        k = np.random.randint(0,2)
        true_label.append(k+1)
        for j in range(worker_num):
            th = np.random.rand()
            if th < alpha[j, k, 0]:
                label[i,j] = 1
            else:
                label[i,j] = 2
    label_mat = np.asmatrix(label)
    gs = InferModule.GibbsSampling(task_num, worker_num, 2)
    acc = gs.test(label_mat, true_label)
    print(acc,"\t",gs.ex_accuracy)
'''


def assign_task(task_num: int, worker_num: int, worker_num_per_task: int):
    """Assign tasks to workers"""
    task_assign_table = [ [] for _ in range(worker_num)]
    worker_id = 0
    for i in range(task_num):
        for k in range(worker_num_per_task):
            # print(worker_id)
            task_assign_table[worker_id].append(i)
            worker_id += 1
            if worker_id >= worker_num:
                worker_id = 0
    return task_assign_table


label = np.zeros(shape=(task_num, worker_num+1), dtype=int)
task_assign_table = assign_task(task_num, worker_num, 3)
true_label = np.random.randint(1, 3, task_num)
for j in range(worker_num):
    for i in task_assign_table[j]:
        label[i, j] = true_label[i]
for i in range(task_num):
    label[i, -1] = 0# true_label[i]

'''
# Lazy Strategy - fully random report

for i in task_assign_table[0]:
    label[i, 0] = np.random.randint(1, 3)
'''

'''
# Lazy Strategy - report '1'
for i in task_assign_table[0]:
    label[i, 0] = 1
'''

'''
# Lazy Strategy - report '2'
for i in task_assign_table[0]:
    label[i, 0] = 2
'''

'''
# Adversarial Strategy
for i in task_assign_table[0]:
    label[i, 0] = 3 - true_label[i]
'''

# print(task_assign_table)
# print(label)

mech = MechModule.DG13()
mech.set([1.0])
p = mech.pay(np.asmatrix(label))
print(p)






