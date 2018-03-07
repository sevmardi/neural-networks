#%% task 1 Import
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from numpy import genfromtxt
import matplotlib.pyplot as plt
import statistics

#%% read and init
train_in = genfromtxt('./data/train_in.csv', delimiter=',')
train_out = genfromtxt('./data/train_out.csv', delimiter=',')
test_in = genfromtxt('./data/test_in.csv', delimiter=',')
test_out = genfromtxt('./data/test_out.csv', delimiter=',')

C = [[] for _ in range(10)]
centerC = [[] for _ in range(10)]
dist = [[] for _ in range(10)]
radiusC = list(range(10))
dists = []

#%% task 1: Get clouds from training data
cnt = 0
for d in train_out :
    C[int(d)].append(train_in[cnt])
    cnt = cnt + 1


#%% Get centers and radius
dists = []
for d in range(10):
    centerC[d] = np.array(C[d]).mean(axis=0)
    radius = 0
    for i in range(len(C[d])):
        dis = distance.euclidean(centerC[d] , C[d][i])
        dists.append(dis)
        if dis > radius:
            radius = dis
    radiusC[d] = radius
radiusC


#%% task 1: Get a matrix of euclid dists between the 10 clouds
dist = [[] for _ in range(10)]
for i in range(10):
    dist[i] = [[] for _ in range(10)]
    for j in range(10):
        dist[i][j] = distance.euclidean(centerC[i] , centerC[j])


#%% task 2 train:
digits = []
for i in range(len(train_in)):
    distances = []
    for c in range(len(centerC)):
        distances.append(distance.euclidean(centerC[c] , train_in[i]))
    digits.append(distances.index(min(distances)))

print(confusion_matrix(digits, train_out))
(digits == train_out).sum() / len(train_out)

#%% task 2 test:
digits = []
for i in range(len(test_in)):
    distances = []
    for c in range(len(centerC)):
        distances.append(distance.euclidean(centerC[c] , test_in[i]))
    digits.append(distances.index(min(distances)))

print(confusion_matrix(digits, test_out))
(digits == test_out).sum() / len(test_out)
