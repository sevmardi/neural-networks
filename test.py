# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 22:29:52 2017

@author: elara
"""

import numpy as np
import pandas as pd
import os

train_in = pd.read_csv('data/train_in.csv', sep=',', header=None)
train_out = pd.read_csv('data/train_out.csv', sep=',', header=None)
test_in = pd.read_csv('data/test_in.csv', sep=',', header=None)
test_out = pd.read_csv('data/test_out.csv', sep=',', header=None)

# Prepare the data
#train = pd.concat([train_out, pd.DataFrame([1]*train_in.shape[0]) ,train_in],axis = 1)
#test = pd.concat([test_out, pd.DataFrame([1]*test_in.shape[0]) ,test_in],axis = 1)
train_in = pd.concat([pd.DataFrame([1]*train_in.shape[0]) ,train_in],axis = 1) # 加入截距项
test_in = pd.concat([pd.DataFrame([1]*test_in.shape[0]) ,test_in],axis = 1)

## task4

# 初始化-----------------------------------------------------------------------
train_n = train_in.shape[0]
dim = train_in.shape[1]
nodes = len(train_out.drop_duplicates())
train_in.columns = range(0, dim)
test_in.columns = range(0, dim)


iter_num = 0 # 迭代次数
falsepos = 1 # 假正数
falseneg = 1 # 假负数
train_out = train_out.astype(int)

# Initialize weight
weight = pd.DataFrame(np.zeros([dim,nodes]))

# Generate weights randomly
#for i in range(nodes):
#    weight.iloc[:,i] = np.random.uniform(-1, 1, size = dim) 
    
# 迭代（don't run)-------------------------------------------------------------
while (((falsepos != 0) | (falseneg != 0)) & (iter_num <= 2)): #进入条件，仍有错分，且迭代次数不超过10000
    y = pd.DataFrame(np.dot(train_in , weight)) # 更新线性函数值    
    y[y>=0]=1 # 更新预测值
    y[y<0]=0 
    y = y.astype(int)
    # 重置不是正确节点却激活数量
    falsepos = 0
    # 重置是正确节点却没激活数量
    falseneg = 0
    for i in range(train_n): 
        #i = 0
        for j in range(nodes):
            #j = 6
            # 不是正确分类的节点, 且节点被激活
            if j != train_out.iloc[i,0] and y.iloc[i,j]==1:
                falsepos = falsepos + 1 
                # w = w - x
                weight.iloc[:,j] = weight.iloc[:,j] - train_in.iloc[i,:]
                #j=6
                # 是正确分类的节点，但是节点没激活
            if j == train_out.iloc[i,0] and y.iloc[i,j]==0:
                falseneg = falseneg + 1
                # w = w + x
                weight.iloc[:,j] = weight.iloc[:,j] + train_in.iloc[i,:]
    iter_num = iter_num + 1
    print iter_num, falsepos, falseneg

# 预测-------------------------------------------------------------------------
weight = pd.read_csv('result_weight.csv', sep=',',header=None)
y = pd.DataFrame(np.dot(train_in , weight))
y[y>=0]=1
y[y<0]=0 

# 真实分类矩阵
yt = pd.DataFrame(np.zeros([train_n,nodes]))
for i in range(train_n):
    for j in range(nodes):
        if j != train_out.iloc[i,0]:
            yt.iloc[i,j] = 0
        else:
            yt.iloc[i,j] = 1
# 验证是否有错误        
d = (y != yt)
np.sum(d.apply(lambda x: x.sum(), axis = 1))              

##############################################################################
# 测试集
test_n = test_in.shape[0]
test_y_score = pd.DataFrame(np.dot(test_in , weight))
test_y_pre = pd.DataFrame([0]*test_n)
for i in range(test_n):
    test_y_pre.iloc[i,0] = test_y_score.iloc[i,:].idxmax()

np.sum(test_y_pre == test_out) / test_n # 87.8%

