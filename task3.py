import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import warnings
from timeit import default_timer as timer


warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(threshold=np.nan)

# start_time = timer()

# training data
train_in = np.genfromtxt("data/train_in.csv", delimiter=",")
test_in = np.genfromtxt("data/test_in.csv", delimiter=",")
# label values
train_out = np.genfromtxt("data/train_out.csv")
test_out = np.genfromtxt("data/test_out.csv")

# create ordered dict to categorize the digits with
digits = OrderedDict(('list_' + str(i), []) for i in range(10))
address_dict = OrderedDict(("list_" + str(i), []) for i in range(10))
digits_test = OrderedDict(("list_" + str(i), []) for i in range(10))
address_dict_test = OrderedDict(("list_" + str(i), []) for i in range(10))

for i in range(10):
    #trainset
    for j in range(len(train_out)):
        if train_out[j] == i:
            digits["list_" + str(i)].append(train_in[j, :])
            address_dict["list_" + str(i)].append(j)
    # testset
    for k in range(len(test_out)):
        if test_out[k] == i:
            digits_test["list_" + str(i)].append(test_in[k, :])
            address_dict_test["list_" + str(i)].append(k)

# means/centers as (10,256)-array
centers = np.zeros((10, 256))
for i in range(10):
    centers[i, :] = np.mean(digits["list_" + str(i)], axis=0)

# number of points that belong to C_i, n_i
for i in range(10):
    print("number of " + str(i) + "s: " + str(len(digits["list_" + str(i)])))

# calculate radii
raddi = np.zeros((10, 1))
for i in range(10):
    radius = 0
    for point in digits["list_" + str(i)]:
        new_radius = np.linalg.norm(point - centers[i, :])
        if new_radius >= radius:
            radius = new_radius
    raddi[i] = radius

# create a distance matrix between centers
centers_dist = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        centers_dist[i, j] = pairwise_distances(
            centers[i, :], centers[j, :], metric='euclidean')

# Task 3
digits0 = np.array(digits["list_0"])
test0 = np.array(digits_test["list_0"])

digits6 = np.array(digits["list_6"])
test6 = np.array(digits_test["list_6"])
