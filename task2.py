import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import OrderedDict
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
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


#TASK 2 

# train_classified = []
# for i in range(len(train_in)):
#     distances = []
#     for j in range(len(centers)):
#         distances.append(distance.euclidean(centers[j], train_in[i]))
#     train_classified.append(distances.index(min(distances)))

# # create confusion matrix for training data
# conf_matrix_train = confusion_matrix(train_out, train_classified)


# #calcute the correctly classified digits
# correct_rate_train = np.zeros(10)
# for i in range(10):
#     correct_rate_train[i] = float(conf_matrix_train[i,i])/np.sum(conf_matrix_train[i,:])

# print("correct rate of training data")
# print(correct_rate_train)


test_classified = []

# create confusion matrix for test data
for i in range(len(test_out)):
    distances = []
    for j in range(len(centers)):
        distances.append(distance.euclidean(centers[j], test_in[i]))
    test_classified.append(distances.index(min(distances)))


conf_matrix_test = confusion_matrix(test_out, test_classified)

correct_rate_test = np.zeros(10)
for i in range(10):
    correct_rate_test[i] = float(conf_matrix_test[i,i])/np.sum(conf_matrix_test[i,:])

print("correct rate on testing data")
print(correct_rate_test)











def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class_name = np.array([0,1,2,3,4,5,6,7,8,9])
np.set_printoptions(precision=2)

#train confusion matrix
plt.figure()
plot_confusion_matrix(conf_matrix_test, classes=class_name, title = 'Confusion matix training set')
# plt.savefig("train_euclidean.png")
plt.show()



