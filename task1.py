import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, OrderedDict
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


# Analyze distances between images
# train_in = pd.read_csv('train_in.csv', sep=',',header=None)
# train_out = pd.read_csv('train_out.csv', sep=',',header=None)

train_in = np.genfromtxt("data/train_in.csv", delimiter=",")
train_out = np.genfromtxt("data/train_out.csv")

####################################################################

d = OrderedDict(("list_" + str(i), []) for i in range(10))
for i in range(10):
	for j in range(len(train_out)):
		if train_out[j] == i:
			d["list_" + str(i)].append(train_in[j, :])

#means/centers as (10, 256)
centers = np.zeros((10,256))
for i in range(10):
	centers[i, :] = np.mean(d["list_" + str(i)], axis=0)


radii = np.zeros((10,1))
for i in range(10):
	radius = 0
	for point in d["list_" + str(i)]:
		new_radius = np.linalg.norm(point -  centers[i,:])
		if new_radius >= radius:
			radius = new_radius
	radii[i] = radius


distance_matrix = np.zeros((10,10))

for i in range(10):
	for j in range(10):
		distance_matrix[i, j] = np.around((np.linalg.norm(centers[i, :])), decimals=2)

for i in range(10):
	print(np.around(np.mean(distance_matrix[i,:]), decimals=2))

##############################################################3
# train_in.head()
# train_out.head()

# # prepare the data
# train = pd.concat([train_out, train_in], axis=1)
# train.columns = range(0, 257)

# # test = pd.concat([test_out, test_in], axis=1)
# # test.columns = range(0, 257)


# ts1 = train.groupby(train_out[0])

# # Calculate the centroid of the 10 classes.
# cal_center = train.groupby(train_out[0]).mean()

# # calculate the distance from the vector that represents this image to
# # each of the 10 centers;
# radius = [[]] * 10

# for i in range(0, 10):
#     seq = (train_in[train_out[0] == i] - cal_center.iloc[i, :])**2
#     radius[i] = np.sum(seq, axis=1).max()**(0.5)

# n = Counter(train_out[0])

# # Distance between 10 centers of classes
# dist_cen = pdist(cal_center, 'euclidean')
# dist_mat = squareform(dist_cen)

# print("---------------Distances between each centers-------------")
# print(dist_mat)




