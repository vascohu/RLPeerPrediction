import numpy as np
import RLModule
import InferModule
import MechModule
import CrowdModule
import time




# Basic parameters
task_num = 100
worker_num = 10
num_true_label = 0
T = 100


# The incentive mechanism
mech = MechModule.DG13()

# The crowd market
mkt = CrowdModule.CrowdMarket(task_num, worker_num, mech)

# The inference
infer = InferModule.GibbsSampling(task_num, worker_num, 2, num_true_label)

# The RL module
rl = RLModule.GpSarsa(2*worker_num)


for i in range(T):
    # Get the action
    a = rl.decide()
    # Set the mechanism
    mech.set([a])
    # Collect the label matrix
    label_mat = mkt.get_label_mat(num_true_label)
    true_label = mkt.get_true_label()
    # Decide the payment
    reward_vec = mech.pay(label_mat)
    pay = np.sum(reward_vec)
    # Inference
    acc = infer.test(label_mat, list(true_label))
    r = infer.reward(pay)
    s = infer.belief
    print("Action: ", a, "\t Reward: ", r)
    print(acc, '\t', infer.ex_accuracy)
    # Observation
    rl.observe(a, r, s)

"""
# Change the incentive level
mech.B = 1

# The label mat and true labels
label_mat = mkt.get_label_mat(num_true_label)
true_label = mkt.get_true_label()

# The inference
acc = infer.test(label_mat, list(true_label))
print(acc, '\t', infer.ex_accuracy)
"""

'''
# Change the incentive level
mech.B = 0.5

# The label mat and true labels
label_mat = mkt.get_label_mat(num_true_label)
true_label = mkt.get_true_label()

# The inference
acc = infer.test(label_mat[:, 0:-1], list(true_label))
print(acc, '\t', infer.ex_accuracy)
'''










