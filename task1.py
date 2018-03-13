import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import Counter
from collections import OrderedDict
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

# for i in range(10):
# 	print("Number of "  + str(i) + "s: " + str(len(d["list_" + str(i)])))

#calculate the radius
radii = np.zeros((10,1))
for i in range(10):
	radius = 0
	for point in d["list_" + str(i)]:
		new_radius = np.linalg.norm(point -  centers[i,:])
		if new_radius >= radius:
			radius = new_radius
	radii[i] = radius


# print (np.around(radii, decimals = 2))

distance_matrix = np.zeros((10,10))

for i in range(10):
	for j in range(10):
		
		print ("Distance (" + str(i) + ", " + str(j) + ") = " + str(np.linalg.norm(centers[i,:]-centers[j,:])))
		# distance_matrix[i, j] = np.around((np.linalg.norm(centers[i, :])), decimals=2)

# for i in range(10):
# 	print(np.around(np.mean(distance_matrix[i,:]), decimals=2))


# for i in range(10):
# 	print("\\textbf{" + str(i)+ "}&")
# for i in range(10):
# 	print ("\\textbf{" + str(i)+ "}&" + '&'.join(str(p) for p in list(distance_matrix[i,:])) + "\\\\")
# 	print ("\\hline")
