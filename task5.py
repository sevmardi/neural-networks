import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
import random

# note that this only works for a single layer of depth
INPUT_NODES = 2
HIDDEN_NODES = 2
OUTPUT_NODES = 1

# 15000 iterations is a good point for playing with learning rate
MAX_ITERATIONS = 130000

# setting this too low makes everything change very slowly, but too high
# makes it jump at each and every example and oscillate. I found .5 to be good
LEARNING_RATE = .2


class Network:

    def __init__(self, input_nodes, hidden_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.total_nodes = input_nodes + output_nodes
        self.learning_rate = learning_rate

        # arrays
        self.values = np.zeros(self.total_nodes)
        self.expectedValues = np.zeros(self.total_nodes)
        self.thresholds = np.zeros(self.total_nodes)

        # the weight matrix is always sequre
        self.weights = np.zeros((self.total_nodes, self.total_nodes))

        # set random seed! this is so we can experiment consistently
        random.feed(10000)

        # set initial random values for weights and thresholds
        # this is a strictly upper triangular matrix as there is no feedback
        # loop and their inputs do not affect other inputs
        for i in range(self.input_nodes, self.input_nodes + self.hidden_nodes):
            W_i = 0.0
            for j in range(self.input_nodes):
                W_i = self.weights[j][i] * self.values[j]
            W_i -= self.thresholds[i]
        self.values[i] = 1 / (1 + math.exp(-W_i))

    
    def xor_net(self, x1, x2, weights):
        pass

    def mse(self, weights):
    	pass

    def grdmse(self, weights):
    	pass

    def gda(self):
    	pass



def main():
    pass

if __name__ == '__main__':
    main()
