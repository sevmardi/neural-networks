import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer


# Analyze distances between images
start_time = timer()

train_in = pd.read_csv('data/train_in.csv', sep=',', header=None)
train_out = pd.read_csv('data/train_out.csv', sep=',', header=None)
test_in = pd.read_csv('data/test_in.csv', sep=',', header=None)
test_out = pd.read_csv('data/test_out.csv', sep=',', header=None)

# train_in = np.genfromtxt("data/train_in.csv", delimiter=",")
# train_out = np.genfromtxt("data/train_out.csv", delimiter=",")
# test_in = np.genfromtxt("data/test_in.csv", delimiter=",")
# test_out = np.genfromtxt("data/test_out.csv", delimiter=",")

elapsed = timer()
# print("Run time (Seconds): " + str(elapsed - start_time))

# train_in.head()
# train_out.head()

# prepare the data
train = pd.concat([train_out, train_in], axis=1)
train.columns = range(0, 257)
test = pd.concat([test_out, test_in], axis=1)
test.columns = range(0, 257)


##########
# Task 1 #
# Analyze distances between images
##########

ts1 = train.groupby(train_out[0])

# Calculate the centroid of the 10 classes.
cal_center = train.groupby(train_out[0]).mean()

# calculate the distance from the vector that represents this image to
# each of the 10 centers;
radius = [[]] * 10

for i in range(0, 10):
    seq = (train_in[train_out[0] == i] - cal_center.iloc[i, :])**2
    radius[i] = np.sum(seq, axis=1).max()**(0.5)

n = Counter(train_out[0])

# Distance between 10 centers of classes
dist_cen = pdist(cal_center, 'euclidean')
dist_mat = squareform(dist_cen)

# print("---------------Distances between each centers-------------")
# print(dist_mat)

##########
# Task 2 # Take #1
# Implement and evaluate the simplest classifier
##########

# out_label = []

# for i in range(0, 1707):
#     seq = (train_in.loc[i, :] - cal_center) ** 2
#     dist = np.sum(seq, axis=1) ** (0.5)
#     out_label.append(np.argmin(dist))


# out_label_test = []

# for i in range(0, 1000):
#     seq = (test_in.loc[i, :] - cal_center) ** 2
#     out_label_test.append(np.argmin(dist))

# Task2 - take #2
train_pre = np.empty(len(train_out))
test_pre = np.empty(len(test_out))

# for i in range(len(train_in)):
# 	current_dist = pairwise_distances(centers, train_in[i], metric='cosine')
# 	train_pre = np.argmin(current_dist)

conf_matrix_train = confusion_matrix(train_out, train_pre)


# calculate the correctly classified digits
correct_rate_train = np.zeros(10)
for i in range(10):
    correct_rate_train[i] = float(
        conf_matrix_train[i, i]) / np.sum(conf_matrix_train[i, :])
print("Correct rate of training")
print(correct_rate_train)


# create CM for test data

for i in range(len(test_out)):
    current_dist = pairwise_distances(centers, test_in[i], metric='cosine')
    test_pre[i] = np.argmin(current_dist)

conf_matrix_test = confusion_matrix(test_out, test_pre)

correct_rate_test = np.zeros(10)
for i in range(10):
    correct_rate_test[i] = float(
        conf_matrix_test[i, i]) / np.sum(conf_matrix_test[i, :])
print("correct rate on testing data")

print(correct_rate_test)


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
	pass

