
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


#%%

n_nodes = 10 # one per digit 0-9
n_inputs = 256 # 256, one for each pixel in a digit intensity vector + 1 for bias
eta = 0.5
iterations = 2

weights = np.random.rand(n_inputs)
errors = np.zeros(10)
layer = np.zeros(n_nodes)

classification = np.zeros(len(train_out))



def train():
    for i in range(iterations):
        #err = 0
        for idx, img in enumerate(train_in):
            target = target_output(int(train_out[idx]))
            for n in range(n_nodes):
                layer[n] = np.sum(np.multiply(weights[n], img))

                errors[n] =  target[n] - layer[n]

                learn(img, layer[n], errors[n])
            classification[idx] = predict(layer)
        print((classification == train_out).sum())
        print(layer)
        print(weights)

def test():
    for idx, img in enumerate(test_in):
        for n in range(n_nodes):
            layer[n] = np.sum(np.multiply(weights[n], img))
        #if(test_out(i) )

train()

#%%
print(classification)
#print(output)

#%%


def target_output(idx):
    out = np.zeros(n_nodes)
    out[idx] = 1
    return out


def learn(inp, out, err):
    #y = 0
    #if (err < 0): y = 0
    #else: y = 1
    for i in range(10):
        print (eta * inp[i] * err)
        weights[i] += (eta * inp[i] * err)

def predict(layer):
    return np.argmax(layer)
