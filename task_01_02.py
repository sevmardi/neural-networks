import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.metrics import confusion_matrix


train_in = pd.read_csv('data/train_in.csv', sep=',', header=None)
train_out = pd.read_csv('data/train_out.csv', sep=',', header=None)
test_in = pd.read_csv('data/test_in.csv', sep=',', header=None)
test_out = pd.read_csv('data/test_out.csv', sep=',', header=None)

train_in.head()
train_out.head()
test_in.head()
test_out.head()


# Prepare the data
train = pd.concat([train_out, train_in], axis=1)
train.columns = range(0, 257)

test = pd.concat([test_out, test_in], axis=1)
test.columns = range(0, 257)

##########
# Task 1 #
##########
ts1 = train.groupby(train_out[0])

# Calculate the centroid of the 10 classes.
centroids = train.groupby(train_out[0]).mean()
radius = [[]] * 10
for i in range(0, 10):
    squared = (train_in[train_out[0] == i] - centroids.iloc[i, :])**2
    radius[i] = np.sum(squared, axis=1).max()**(0.5)

dist_cen = pdist(centroids, 'euclidean')
dis_mat = squareform(dist_cen)

print(dist_cen)

##########
# Task 2 #
#Implement and evaluate the simplest classifier
##########
##########
out_label = []
for i in range(0, 1707):
    squared = (train_in.loc[i, :] - centroids)**2
    dist = np.sum(squared, axis=1) ** (0.5)
    out_label.append(np.argmin(dist))

# print(out_label)
out_label_test = []
for i in range(0, 1000):
    squared = (train_in.loc[i, :] - centroids)**2
    dist = np.sum(squared, axis=1) ** (0.5)
    out_label_test.append(np.argmin(dist))

# print(out_label_test)
