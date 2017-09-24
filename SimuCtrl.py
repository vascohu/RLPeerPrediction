import numpy as np
import RLModule
import InferModule

worker_num = 5
task_num = 100

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




