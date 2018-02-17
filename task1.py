import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


# Analyze distances between images

train_in = pd.read_csv('data/train_in.csv', sep=',', header=None)
train_out = pd.read_csv('data/train_out.csv', sep=',', header=None)
test_in = pd.read_csv('data/test_in.csv', sep=',', header=None)
test_out = pd.read_csv('data/test_out.csv', sep=',', header=None)


# train_in.head()
# train_out.head()

# prepare the data
train = pd.concat([train_out, train_in], axis=1)
train.columns = range(0, 257)

test = pd.concat([test_out, test_in], axis=1)
test.columns = range(0, 257)

ts1 = train.groupby(train_out[0])
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
