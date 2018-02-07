import numpy as np
import RLModule
import InferModule
import MechModule
import CrowdModule
import sys
import time
import pickle



# Basic parameters
task_num = 100
worker_num = 10
num_true_label = 0
T = 200
EP = 25

# np.random.seed(0)
'''This area for the one-step test of our mechanism'''
'''
CrowdModule.Task.p=0.5
CrowdModule.TRWorker.P_H = 0.5
num_exp = 1000
#payment = np.zeros(shape=(8,num_exp))
#file_name = 'VS_Bayes_High_True8.pkl'
# f = open(file_name, 'rb')
# payment2 = pickle.load(f)
# payment[0:18,:] = payment2[0:18,:]
# f.close()
# for i in range(8):
# for i in [18]:
X = np.zeros(shape=(11, num_exp))
Y = np.zeros(shape=(11, num_exp))
Z = np.zeros(shape=(11, num_exp))
pay = np.zeros(shape=(1, num_exp))
file_name = 'NewExp2.pkl'
for i in range(11):
    CrowdModule.TRWorker.P_H = 0.5 + i*0.05
    #CrowdModule.Task.p = 0.05 + 0.05 * i
    print(CrowdModule.TRWorker.P_H)
    for t in range(num_exp):
        if (t+1)%100 == 0:
            print(t+1)
        else:
            print((t+1), ', ', end='', flush=True)
        mech = MechModule.DG13()
        mkt = CrowdModule.CrowdMarket(task_num, worker_num, mech)
        infer = InferModule.GibbsSamplingSC(task_num, worker_num)
        #label_mat = mkt.get_label_mat(0)
        label_mat = mkt.get_label_mat_NTL()
        #pay = mech.pay(label_mat)
        # print(np.sum(pay[1][0])/90)
        # print(label_mat)
        # infer.infer(label_mat)
        # pay[0, t] = infer.p_vec[0]
        # print(pay[0, t])
        #payment[i, t] = infer.p_vec[0]
        #revenue[i, t] = infer.ex_accuracy
        Y[i, t] = infer.test(label_mat, mkt.get_true_label())
        X[i, t] = infer.ex_x
        Z[i, t] = infer.p_vec[0]
        #print(infer.p_vec)
        #print(infer.x_vec)
        #payment[i, t] = np.sum(pay[1][0])/90

f = open(file_name, 'wb')
#pickle.dump(payment, f)
#pickle.dump(revenue, f)
pickle.dump(X, f)
pickle.dump(Y, f)
pickle.dump(Z, f)
#pickle.dump(pay, f)
f.close()
'''


# The incentive mechanism
mech = MechModule.BayesMech()
#mech = MechModule.DG13()#

# The crowd market
mkt = CrowdModule.CrowdMarket(task_num, worker_num, mech, sys.argv[1])

# The inference
infer = InferModule.GibbsSamplingSC(task_num, worker_num)

# The RL module
# rl = RLModule.TOSarsa(EP)
rl = RLModule.EpSGPS(EP)
# rl = RLModule.Simple()


# label_mat = mkt.get_label_mat_NTL()
# true_label = mkt.get_true_label()


# accuracy = infer.test(label_mat, true_label)
#print(true_label)
# print(accuracy)
# print(infer.p_vec)
# print(infer.ex_accuracy)

a = 0
s = 0

rr = []
RR=[]


'''
mech.set([10])
label_mat = mkt.get_label_mat(num_true_label)
(pay, reward_mat) = mech.pay(label_mat)
mkt.evolve(reward_mat)
'''

rl.explore_prob = 0.2

thefile = open('rl_' + sys.argv[1]+'.txt', 'w')

for i in range(T):
    print(">>>>>>>>>>>>Round: %i"% i)
    mkt.worker_init()
    accR = 0
    accRR = 0
    rl.explore_prob *= 0.99
    for t in range(EP):
        #print("Step: ", t+1)
        # Get the action
        if t==0:
            a = rl.decide(start=True)
        else:
            a = rl.decide(start=False)

        # Set the mechanism
        mech.set([a])

        # Decide the payment
        if isinstance(mech, MechModule.DG13):
            label_mat = mkt.get_label_mat(num_true_label)
            true_label = mkt.get_true_label()
            (pay, reward_mat) = mech.pay(label_mat)
            acc = infer.test(label_mat, list(true_label))
        else:
            label_mat = mkt.get_label_mat_NTL()
            true_label = mkt.get_true_label()
            acc = infer.test(label_mat, list(true_label))
            (pay, reward_mat) = mech.pay(label_mat, infer.belief)
        mkt.evolve(reward_mat)
        r = infer.reward(pay)
        accR += r
        accRR += infer.real_reward(acc, pay)
        #s[0] = (np.mean(infer.belief[0::2])*infer.beta[0]+np.mean(infer.belief[1::2])*infer.beta[1])/np.sum(infer.beta)
        s = np.mean(infer.belief)
        # s[0] = infer.ex_accuracy
        # s[1] = t+1
        # print("Action: ", a, "\t Reward: ", r)
        # print(pay)
        # print(acc, '\t', infer.ex_accuracy)

        #print("State: ", rl.z, "Action: ", a, "Reward: ", r)
        # Observation
        if t==EP-1:
            rl.observe(a, r, s, terminal=True)
        elif t==1:
            rl.observe(a, r, s, start=True)
            # rl.decide(start=False)
        else:
            rl.observe(a, r, s)
    #print(accR)
    #print(accRR)
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










