import math
import numpy as np
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

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.total_nodes = input_nodes + hidden_nodes + output_nodes
        self.learning_rate = learning_rate

        # arrays
        self.values = np.zeros(self.total_nodes)
        self.expectedValues = np.zeros(self.total_nodes)
        self.thresholds = np.zeros(self.total_nodes)

        # the weight matrix is always sequre
        self.weights = np.zeros((self.total_nodes, self.total_nodes))

        # set random seed! this is so we can experiment consistently
        random.seed(10000)

        # set initial random values for weights and thresholds
        # this is a strictly upper triangular matrix as there is no feedback
        # loop and their inputs do not affect other inputs
        for i in range(self.input_nodes, self.total_nodes):
            self.thresholds[i] = random.random() / random.random()
            for j in range(i + 1, self.total_nodes):
                self.weights[i][j] = random.random() * 2

    def mse(self):
        sum_of_errors = 0.0

        for i in range(self.input_nodes + self.hidden_nodes, self.total_nodes):
            error = self.expectedValues[i] - self.values[i]

            # print erorr
            sum_of_errors += math.pow(error, 2)
            output_error_gradient = self.values[
                i] * (1 - self.values[i]) * error

            # now update the weights and thresholds
            for j in range(self.input_nodes, self.input_nodes + self.hidden_nodes):
                delta = self.learning_rate * \
                    self.values[i] * output_error_gradient
                # print(delta)
                self.weights[j][i] += delta
                hidden_error_gradient = self.values[j] * output_error_gradient

                # and then update for the input nodes to hidden nodes
                for k in range(self.input_nodes):
                    delta = self.learning_rate * \
                        self.values[k] * hidden_error_gradient
                    self.weights[k][j] += delta

                # update the thresholds for the hidden nodes
                delta = self.learning_rate * -1 * hidden_error_gradient
                # print(delta)
                self.thresholds[j] += delta

            delta = self.learning_rate * -1 * output_error_gradient
            self.thresholds[i] += delta

        return sum_of_errors

    def grdmse(self, weights):
        pass

    def gda(self):
        # update the hidden node
        for i in range(self.input_nodes, self.input_nodes + self.hidden_nodes):
            W_i = 0.0
            for j in range(self.input_nodes):
                W_i += self.weights[j][i] * self.values[j]
            W_i -= self.thresholds[i]
            self.values[i] = self.__sigmoid(W_i)

        # update the output node
        for i in range(self.input_nodes + self.hidden_nodes, self.total_nodes):
            W_i = 0.0
            for j in range(self.input_nodes, self.input_nodes + self.hidden_nodes):
                W_i += self.weights[j][i] * self.values[j]
            W_i -= self.thresholds[i]
            self.values[i] = self.__sigmoid(W_i)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)


class NetworkMaker:

    def __init__(self, network):
        self.counter = 0
        self.network = network

    def set_xor(self, weights):
        if weights == 0:
            self.network.values[0] = 1
            self.network.values[1] = 1
            self.network.expectedValues[4] = 0
        elif weights == 1:
            self.network.values[0] = 0
            self.network.values[1] = 1
            self.network.expectedValues[4] = 1
        elif weights == 2:
            self.network.values[0] = 1
            self.network.values[1] = 0
            self.network.expectedValues[4] = 1
        else:
            self.network.values[0] = 0
            self.network.values[1] = 0
            self.network.expectedValues[4] = 0

    def set_training_data(self):
        self.set_xor(self.counter % 4)
        self.counter += 1


def main():
    net = Network(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
    sample = NetworkMaker(net)

    for i in range(MAX_ITERATIONS):
        sample.set_training_data()
        net.gda()
        error = net.mse()

        if i > (MAX_ITERATIONS - 5):
            output = (net.values[0], net.values[1], net.values[4], net.expectedValues[4], error)
            print(output)

    print(net.weights)
    print(net.thresholds)



if __name__ == '__main__':
    main()
