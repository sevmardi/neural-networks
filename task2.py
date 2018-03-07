import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances

train_in = np.genfromtxt("data/train_in.csv", delimiter=",")
test_in = np.genfromtxt("data/test_in.csv", delimiter=",")
# label values
train_out = np.genfromtxt("data/train_out.csv")
test_out = np.genfromtxt("data/test_out.csv")


# train_in = pd.read_csv('data/train_in.csv', sep=',',header=None)
# train_out = pd.read_csv('data/train_out.csv', sep=',',header=None)
# test_in = pd.read_csv('data/test_in.csv', sep=',',header=None)
# test_out = pd.read_csv('data/test_out.csv', sep=',',header=None)


# Prepare the data
# train = pd.concat([train_out, train_in], axis = 1)
# train.columns = range(0, 257)

# test = pd.concat([test_out, test_in], axis = 1)
# test.columns = range(0, 257)


# create ordered dictionary with (list_#digit: vectors)
d = OrderedDict(("list_" + str(i), []) for i in range(10))
for i in range(10):
    for j in range(len(train_out)):
        if train_out[j] == i:
            d["list_" + str(i)].append(train_in[j, :])

# means/centers as (10,256)-array
centers = np.zeros((10, 256))
# centers = train.groupby(train_out[0]).mean()

for i in range(10):
    centers[i, :] = np.mean(d["list_" + str(i)], axis=0)

# for i in range(10):
#     print("number of " + str(i) + "s: " + str(len(d["list_" + str(i)])))

# calculate radii
radii = np.zeros((10, 1))
for i in range(10):
    radius = 0
    for point in d["list_" + str(i)]:
        new_radius = np.linalg.norm(point - centers[i, :])
        if new_radius >= radius:
            radius = new_radius
    radii[i] = radius


# create distance matrix between centers
centers_dist = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        centers_dist[i, j] = np.linalg.norm(centers[i, :] - centers[j, :])
# print("Distances between each centers")
# print(centers_dist)

train_pre = np.empty(len(train_out))
test_pre = np.empty(len(test_out))

train_classified = []
for i in range(len(train_in)):
    distances = []
    for c in range(len(centers)):
        distances.append(distance.euclidean(centers[c], train_in[i]))
    train_classified.append(distances.index(min(distances)))
    #train_pre[i] = np.argmin(current_dist)

print(confusion_matrix(train_classified, train_out))
(train_classified == train_out).sum() / len(train_out)

#confusion_matrix_test = confusion_matrix(test_out, train_classified)
correct_rate_train = np.zeros(10)
conf_matrix_train = confusion_matrix(train_out, train_classified)


# # calculate the correctly classified digits
# correct_rate_train = np.zeros(10)
# for i in range(10):
#     correct_rate_train[i] = float(
#         conf_matrix_train[i, i]) / np.sum(conf_matrix_train[i, :])
# print("Correct rate of training")
# print(correct_rate_train)


# create confusion matrix for test data
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
