import numpy as np
import pandas as pd
import time

start = time.time()


train_in = pd.read_csv('data/train_in.csv', sep=',', header=None)
train_out = pd.read_csv('data/train_out.csv', sep=',', header=None)
test_in = pd.read_csv('data/test_in.csv', sep=',', header=None)
test_out = pd.read_csv('data/test_out.csv', sep=',', header=None)


train_in = pd.concat(
    [pd.DataFrame([1] * train_in.shape[0]), train_in], axis=1)  # 加入截距项
test_in = pd.concat([pd.DataFrame([1] * test_in.shape[0]), test_in], axis=1)

# task4

train_n = train_in.shape[0]
dim = train_in.shape[1]

nodes = len(train_out.drop_duplicates())
train_in.columns = range(0, dim)
test_in.columns = range(0, dim)


iter_num = 0
runs = 10

falsepos = 1
falseneg = 1


train_out = train_out.astype(int)


# init weight
weight = pd.DataFrame(np.zeros([dim, nodes]))

while (((falsepos != 0) | (falseneg != 0)) & (iter_num <= 5)):
    y = pd.DataFrame(np.dot(train_in, weight))

    y[y >= 0] = 1
    y[y < 0] = 0

    falsepos = 0
    falseneg = 0

    for i in range(train_n):
        for j in range(nodes):
            if j != train_out.iloc[i, 0] and y.iloc[i, j] == 1:
                falsepos += 1
                weight.iloc[:, j] = weight.iloc[:, j] - train_in.iloc[i, :]
            if j == train_out.iloc[i, 0] and y.iloc[i, j] == 0:
                falseneg += 1
                weight.iloc[:, j] = weight.iloc[:, j] + train_in.iloc[i, :]
    iter_num += 1

    print(iter_num, falsepos, falseneg)

weight = pd.read_csv('result_weight.csv', sep=',', header=None)
y = pd.DataFrame(np.dot(train_in, weight))
y[y >= 0] = 1
y[y < 0] = 0

yt = pd.DataFrame(np.zeros([train_n, nodes]))
for i in range(train_n):
    for j in range(nodes):
        if j != train_out.iloc[i, 0]:
            yt.iloc[i, j] = 0
        else:
            yt.iloc[i, j] = 1

d = (y != yt)
np.sum(d.apply(lambda x: x.sum(), axis=1))

#test 
test_n = test_in.shape[0]
test_y_score = pd.DataFrame(np.dot(test_in, weight))
test_y_pre = pd.DataFrame([0] * test_n)
for i in range(test_n):
    test_y_pre.iloc[i, 0] = test_y_score.iloc[i, :].idxmax()
np.sum(test_y_pre == test_out) / test_n
