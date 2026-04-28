"""

First attempt at creating a generic Neural Network
with backpropagation implementation from first principles.
(i.e., without looking at the code in the book)

- sigmoid activation function
- MSE cost function
- object-oriented approach

"""

import numpy as np

class Network:
    def __init__(self, dimensions):
        ... # Initialise network

    def calc_network(self, inputs):
        ... # Feedfoward from input nodes to output nodes

    def calc_cost(self, expected):
        ... # Calculate the cost of the network using the MSE cost function

    def train(self, data):
        ... # Train the network given the training data

    def calculate_backprop_gradients(self):
        ... # Calculate how changing each weight and bias in the network
        # affects the overall cost function (backpropagation)

if __name__ == '__main__':
    net = Network(
        # Params
    )
    # [Interact with network]
