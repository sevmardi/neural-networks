import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Task 4 -
# load data
train_in = np.genfromtxt("data/train_in.csv", delimiter=",")
test_in = np.genfromtxt("data/test_in.csv", delimiter=",")
train_out = np.genfromtxt("data/train_out.csv")
test_out = np.genfromtxt("data/test_out.csv")

train_in = np.insert(train_in, 256, 1, axis=1)
test_in = np.insert(test_in, 256, 1, axis=1)


sum_temp = np.zeros(10)


iters = 40
runs = 10

mis_train_all = np.zeros((runs, iters))
mis_test_all = np.zeros((runs, iters))

# number of misclassfied digits for each run
errors = np.zeros(10)

#algorthm.. later to comment
for run in range(runs):
    weights = np.random.random((10, 257))
    mis_train = np.zeros(iters)
    mis_test = np.zeros(iters)
    mis_min = len(test_in)

    opt_iteration = 0

    for i in range(iters):
        # decrease the learning rate over time
        learning_rate = 9 ** (float(-i) / 20 + 1)
        # actual training
        for j in range(len(train_in)):
            for k in range(10):
                sum_temp[k] = np.sum(np.multiply(weights[k], train_in[j]))
            if train_out[j] != np.argmax(sum_temp):
                # increase the weights of target digit
                weights[int(train_out[j])] += np.multiply(train_in[j], learning_rate)
                # decrease the weights of the digit being misclassfied as
                weights[np.argmax(sum_temp)
                        ] -= np.multiply(train_in[j], learning_rate)
                mis_train[i] += 1
    for l in range(len(test_in)):
        for k in range(10):
            sum_temp[k] = np.sum(np.multiply(weights[k], test_in[l]))

        # if theres a miscassified digit
        if test_out[l] != np.argmax(sum_temp):
            mis_test[i] += 1
        # ratchet: save the best mistake rate and weights configuration seen so far
        if mis_test[i] < mis_min:
            opt_weights = np.copy(weights)
            opt_iteration = i
            mis_min = mis_test[i]

    # mistake rates
    mist_train = mis_train / len(train_in)
    mist_test = mis_test / len(test_in)

    errors[run] = mis_test[opt_iteration]
    mis_test_all[run] = mis_test
    mis_train_all[run] = mis_train

    print("The best Iterations is " + str(opt_iteration + 1))
    print("With the best miste rate of " + str(mis_test[opt_iteration]))

print("The avg accuracy is " + str(1 - np.mean(errors)))

print(np.mean(mis_train_all, axis=0), np.std(mis_train_all, axis=1))
print(np.mean(mis_test_all, axis=0), np.std(mis_test_all, axis=0))

# iter_x = np.linspace(0, iters, iters)
# plt.figure()
# axes = plt.gca()
# axes.set_ylim([0, 0.25])
# plt.errorbar(iter_x, np.mean(mis_train_all, axis=0), yerr=np.std(
#     mis_train_all, axis=0), label="training mistake error")
# plt.errorbar(iter_x, np.mean(mis_test_all, axis=0), yerr=np.std(
#     mis_test_all, axis=0), label="test mistake rate")

# plt.legend(loc=0)
# filepathname = 'plots/task4.png'
# plt.savefig(filepathname)
# print('Done!' + ' Check this folder => ' + filepathname)
# plt.show()
