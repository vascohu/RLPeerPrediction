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
T = 200
EP = 25

# np.random.seed(0)

# The incentive mechanism
mech = MechModule.DG13()

# The crowd market
mkt = CrowdModule.CrowdMarket(task_num, worker_num, mech)

# The inference
infer = InferModule.GibbsSampling(task_num, worker_num, 2, num_true_label)

# The RL module
# rl = RLModule.TOSarsa(EP)
# rl = RLModule.EpSGPS(EP)
rl = RLModule.Simple()

a = 0
s = [0, 0]

rr = []
RR=[]

'''
mech.set([10])
label_mat = mkt.get_label_mat(num_true_label)
(pay, reward_mat) = mech.pay(label_mat)
mkt.evolve(reward_mat)
'''

rl.explore_prob = 0.2

thefile = open('ResultS3.txt', 'w')

for i in range(T):
    print(">>>>>>>>>>>>Round: \n", i)
    mkt.worker_init()
    accR = 0
    accRR = 0
    rl.explore_prob *= 0.99
    for t in range(EP):
        print("Step: ", t+1)
        # Get the action
        if t==0:
            a = rl.decide(start=True)
        else:
            a = rl.decide(start=False)

        # Set the mechanism
        mech.set([a])
        # Collect the label matrix
        label_mat = mkt.get_label_mat(num_true_label)
        true_label = mkt.get_true_label()
        # Decide the payment
        (pay, reward_mat) = mech.pay(label_mat)
        mkt.evolve(reward_mat)
        # Inference
        acc = infer.test(label_mat, list(true_label))
        r = infer.reward(pay)
        accR += r
        accRR += infer.real_reward(acc, pay)
        s[0] = (np.mean(infer.belief[0::2])*infer.beta[0]+np.mean(infer.belief[1::2])*infer.beta[1])/np.sum(infer.beta)
        # s[0] = infer.ex_accuracy
        s[1] = t+1
        # print("Action: ", a, "\t Reward: ", r)
        # print(pay)
        # print(acc, '\t', infer.ex_accuracy)

        print("State: ", rl.z, "Action: ", a, "Reward: ", r)
        # Observation
        if t==EP-1:
            rl.observe(a, r, s, terminal=True)
        elif t==1:
            rl.observe(a, r, s, start=True)
            # rl.decide(start=False)
        else:
            rl.observe(a, r, s)
    print(accR)
    print(accRR)
    rr.append(accR)
    RR.append(accRR)
    thefile.write("%s\t%s\n" % (accR, accRR))
    # for (h,r) in zip(rl.Hist, rl.R):
    #    print(h, "  >>>  ",r )

    #print("The reward is ", accR, "\n\n")
print(rr)
print(RR)

thefile.close()

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










