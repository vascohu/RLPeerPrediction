import InferModule
import CrowdModule
import MechModule
import sys

from itertools import product

num_time_steps = 24
task_num = 100
worker_num = 10
num_true_label = 0
action_set = [0.1, 1, 5, 10]

mech = MechModule.BayesMech()
mkt = CrowdModule.CrowdMarket(task_num, worker_num, mech, sys.argv[1])
infer = InferModule.GibbsSamplingSC(task_num, worker_num)

thefile = open("baseline_" + sys.argv[1]+".txt", 'w')
max_cumulative_reward = None

for action_sequence in product([0,1,2,3], repeat=8):
    mkt.worker_init()
    a = None
    accR = 0
    accRR = 0
    for i in range(num_time_steps):
        if i % 3 == 0:
            a = action_set[action_sequence[i//3]]
        mech.set([a])
        label_mat = mkt.get_label_mat_NTL()
        true_label = mkt.get_true_label()
        acc = infer.test(label_mat, list(true_label))
        (pay, reward_mat) = mech.pay(label_mat, infer.belief)
        mkt.evolve(reward_mat)
        r = infer.reward(pay)
        accR += r
        accRR += infer.real_reward(acc, pay)
        thefile.write("%s\t%s\n" % (accR, accRR))
        thefile.flush()
        if max_cumulative_reward == None or accRR > max_cumulative_reward:
            max_cumulative_reward = accRR
thefile.write("max_cumulative_real_reward %s\n" % max_cumulative_reward)
thefile.close()

