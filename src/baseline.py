import InferModule
import CrowdModule
import MechModule
import sys

from itertools import product

num_time_steps = 28
task_num = 100
worker_num = 10
num_true_label = 0
action_set = [0.1, 1, 5, 10]

mech = MechModule.BayesMech()
mkt = CrowdModule.CrowdMarket(task_num, worker_num, mech, sys.argv[1])
infer = InferModule.GibbsSamplingSC(task_num, worker_num)

thefile = open("baseline_" + sys.argv[1]+".txt", 'w')
max_cumulative_real_reward = None
max_cumulative_infered_reward = None

for action_sequence in product([0,1,2,3], repeat=7):
    mkt.worker_init()
    a = None
    accR = 0
    accRR = 0
    for i in range(num_time_steps):
        if i % 4 == 0:
            a = action_set[action_sequence[i//4]]
        mech.set([a])
        label_mat = mkt.get_label_mat_NTL()
        true_label = mkt.get_true_label()
        acc = infer.test(label_mat, list(true_label))
        (pay, reward_mat) = mech.pay(label_mat, infer.p_dist)
        mkt.evolve(reward_mat)
        r = infer.reward(pay)
        accR += r
        accRR += infer.real_reward(acc, pay)
    thefile.write("%s\t%s\n" % (accR, accRR))
    thefile.flush()
    if max_cumulative_real_reward == None or accRR > max_cumulative_real_reward:
        max_cumulative_real_reward = accRR
    if max_cumulative_infered_reward == None or accR > max_cumulative_infered_reward:
        max_cumulative_infered_reward = accR

thefile.write("max_cumulative_infered_reward %s\n" % max_cumulative_infered_reward)
thefile.write("max_cumulative_real_reward %s\n" % max_cumulative_real_reward)
thefile.close()

