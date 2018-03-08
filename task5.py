import copy
import math
import numpy as np
import random
import sys



# 15000 iterations is a good point for playing with learning rate
MAX_ITERATIONS = 130000

# setting this too low makes everything change very slowly, but too high
# makes it jump at each and every example and oscillate. I found .5 to be good
LEARNING_RATE = .2


class Network:

    def xor_net(self, x1, x2, weights):
        """
        
        """
        self.input_nodes = self.__finding_value(x1, x2, weights[0:3])
        self.hidden_nodes = self.__finding_value(x1, x2, weights[3:6])
        self.output_nodes = self.__finding_value(x1, x2, weights)

        return self.output_nodes

    def mse(self, weights):
        """ Calculate mean sequared error

        Parameters
        ------
        Weights: Node weights  


        Return
        ------
        sum_of_errors: mean sequared on 4 input vectors

        """
        sum_of_errors = 0.0
        x_matrix = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        xor_vec = np.array([0, 1, 1, 0])

        for i in range(0, len(xor_vec)):
            sum_of_errors += self.__error_calc(x_matrix[0][i], x_matrix[
                1][i], xor_vec[i], weights)

        return sum_of_errors

    def grdmse(self, weights):
        """  Gradient of mes(weights) 

        Parameters 
        ----------

        Return
        ------
        """
        value = np.zeros(9)

        x_matrix = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        xor_vec = np.array([0, 1, 1, 0])
        partial_dev = 10**-3
        for i in range(0, 9):
            W_i = copy.copy(weights)
            W_i[i] += partial_dev
            value[i] = ((self.mse(W_i) - self.mse(weights)) / partial_dev)

        return value

    def gda(self):
        """ Gradient Descent Algorithm
        Parameters
        ----------
        None 

        Return
        ------
        None

        """
        weights = np.random.rand(9)
        for i in range(MAX_ITERATIONS):
            weights = weights - LEARNING_RATE * self.grdmse(weights)
        print("Error: ", self.mse(weights))

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __finding_value(self, x1, x2, weights):
        return self.__sigmoid(np.sum([x1, x2] * weights[0:2]) + weights[2])

    def __error_calc(self, x1, x2, d, weights):
        return (self.xor_net(x1, x2, weights) - d) ** 2


def main():
    weights = np.random.rand(9)
    weights[2] = 1
    weights[5] = 1
    weights[8] = 1
    net = Network()

    print('init error', net.mse(weights))

    for i in range(MAX_ITERATIONS):
        if i > (MAX_ITERATIONS - 5):
            weights = weights - LEARNING_RATE * net.grdmse(weights)

    print('erorr new state:', net.mse(weights))
    print(net.xor_net(0, 0, weights))
    print(net.xor_net(0, 1, weights))
    print(net.xor_net(1, 0, weights))
    print(net.xor_net(1, 1, weights))


if __name__ == '__main__':
    main()
