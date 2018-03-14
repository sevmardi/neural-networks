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
    # trainset
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
# for i in range(10):
#     print("number of " + str(i) + "s: " + str(len(digits["list_" + str(i)])))

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


dist0 = np.zeros(len(digits0))
dist6 = np.zeros(len(digits6))
dist_test0 = np.zeros(len(test0))
dist_test6 = np.zeros(len(test6))
mis_counts0 = np.zeros(80)
mis_counts6 = np.zeros(80)
mis_test0 = 0.0
mis_test6 = 0.0

for i in range(len(dist0)):
    dist0[i] = pairwise_distances(digits[0], centers[0], metric='euclidean') - pairwise_distances(digits0[6],  metric="euclidean")
    for j in range(80):
        if dist0[i] > -4 + 0.1 * j:
            mis_counts0[j] += 1

misclassified = mis_counts0 + mis_counts6
mis_rate0 = np.copy(mis_counts0) / len(dist0)
mis_rate6 = np.copy(mis_counts6) / len(dist6)
mis_rate = np.copy(misclassified) / (len(dist0) + len(dist6))

#Plot
bins = np.linspace(min(dist0), max(dist6), 60)
plt.hist(dist0, bins, alpha=0.5, label='digit0')
plt.hist(dist6, bins, alpha=0.5, label='digit6')
plt.legen(loc='Upper right')
plt.show()

# plot the mis classified rate according to the classifier position 
mis_x = np.linspace(-4, 4, 80)
opt_classifier = -4 + 0.1 * np.argmin(misclassified)
plt.figure(3)
plt.plot(mis_x, mis_rate0, label="digit0 misclassified rate")
plt.plot(mis_x, mis_rate6, label="digit6 misclassified rare")
plt.plot(mis_x, mis_rate, label="total misclassified rate")
plt.legend(loc=0)
plt.show()

print("The optimal classifier would be at " + str(opt_classifier))
print("In training set the misclassified rate is " + str(min(misclassified) / (len(dist0) + len(dist6))))
print("The misclassified rate of digit 0 is " + str(mis_counts6[np.argmin(misclassified)] / len(dist0)))
print("The misclassified digit 0 and 6 are " + str(mis_counts0[np.argmin(misclassified)] / len(dist6)))

#Applying the classifier on the test dataset 

for i in range(len(test)):
    dist_test0[i] = pairwise_distances(test0[i], centers[0], metric='euclidean') - pairwise_distances(test0[i], centers[6], metric="euclidean")
    if dist_test0[i] > opt_classifier:
        mis_test0 += 1
for i in range(len(test6)):
    dist6[i] = pairwise_distances(test6[i], centers[0], metric='euclidean') - pairwise_distances(test6[i], centers[6], metric="euclidean")
    if dist_test6[i] <= opt_classifier:
        mis_counts6 +=1
print("In test set the misclassified rate of digit 0 is " + str(mis_test6 / len(test0)))
print ("In test set the misclassified rate of digit 6 is " + str(mis_test0 / len(test6)))
print ("In test set the overall misclassified rate is " + str((mis_test0 + mis_test6) / (len(test6) + len(test0))))

bins = np.linspace(min(dist0), max(dist6), 60)
plt.hist(dist0, bins, alpha=0.5, label='digit0')
plt.hist(dist6, bins, alpha=0.5, label='digit6')
plt.legend(loc='upper right')
plt.show()