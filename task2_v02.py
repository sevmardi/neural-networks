import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

train_in = np.genfromtxt("data/train_in.csv", delimiter=",")
test_in = np.genfromtxt("data/test_in.csv", delimiter=",")
# label values
train_out = np.genfromtxt("data/train_out.csv")
test_out = np.genfromtxt("data/test_out.csv")

d = OrderedDict(("list_" + str(i), []) for i in range(10))
for i in range(10):
	for j in range(len(train_out)):
		if train_out[j] == i:
			d["list_" + str(i)].append(train_in[j, :])

centers = np.zeros((10, 256))
for i in range(10):
	centers[i, :] = np.mean(d["list_" + str(i)], axis=0)

# for i in range(10):
# 	print("number of " + str(i) + "s: " + str(len(d["list_" + str(i)])))

radii = np.zeros((10, 1))
for i in range(10):
	radius = 0
	for point in d["list_" + str(i)]:
		new_radius = np.linalg.norm(point-centers[i,:])
		if new_radius >= radius:
			radius = new_radius
	radii[i] = radius


train_pre = np.empty(len(train_out))
test_pre = np.empty(len(test_out))

for i in range(len(train_in)):
	current_dist = pairwise_distances(centers, train_in[i], metric='cosine')
	train_pre[i] = np.argmin(current_dist)

confusion_matrix_test = confusion_matrix(test_out, test_pre)
correct_rate_test = np.zeros(10)
for i in range(10):
	correct_rate_test[i] = float(confusion_matrix_test[i,i]) / np.sum(confusion_matrix_test[i, :])
print("correct rate of testing data ")
print(correct_rate_test)









# Task2 - take #2
# train_pre = np.empty(len(train_out))
# test_pre = np.empty(len(test_out))

# # for i in range(len(train_in)):
# # 	current_dist = pairwise_distances(centers, train_in[i], metric='cosine')
# # 	train_pre = np.argmin(current_dist)

# conf_matrix_train = confusion_matrix(train_out, train_pre)


# # calculate the correctly classified digits
# correct_rate_train = np.zeros(10)
# for i in range(10):
#     correct_rate_train[i] = float(
#         conf_matrix_train[i, i]) / np.sum(conf_matrix_train[i, :])
# print("Correct rate of training")
# print(correct_rate_train)


# # create CM for test data
# for i in range(len(test_out)):
#     current_dist = pairwise_distances(centers, test_in[i], metric='cosine')
#     test_pre[i] = np.argmin(current_dist)

# conf_matrix_test = confusion_matrix(test_out, test_pre)

# correct_rate_test = np.zeros(10)
# for i in range(10):
#     correct_rate_test[i] = float(
#         conf_matrix_test[i, i]) / np.sum(conf_matrix_test[i, :])
# print("correct rate on testing data")

# print(correct_rate_test)
