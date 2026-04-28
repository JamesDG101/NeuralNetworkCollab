"""

First attempt at creating a generic Neural Network
with backpropagation implementation from first principles.
(i.e., without looking at the code in the book)

- sigmoid activation function
- MSE cost function
- object-oriented approach

"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Network:

    def __init__(self, dimensions):
        self.dim = dimensions

        # Empty array for index 0 of weights and biases (as the first layer of the network
        # is just the input nodes -> we don't do anything to these values)
        self.biases  = [np.array([])] + [np.random.randn(j) for j in self.dim[1:]]
        self.weights = [np.array([])] + [np.random.randn(j, k) for j, k in zip(self.dim[1:], self.dim[:-1])]
        self.activations = [np.zeros(j) for j in self.dim]

        # DEBUG
        # print(self.biases)
        # print(self.weights)
        # print(self.activations)

    def calc_network(self, inputs):
        # Feedfoward from input nodes to output nodes

        # Reset all activations to zero, setting first layer to input values
        self.activations = [np.array(inputs)] + [np.zeros(j) for j in self.dim[1:]]

        # For each layer of network
        for l in range(1, len(self.dim)):

            # For each node in layer
            for j in range(self.dim[l]):

                # Start by setting its value to the node's bias
                z = self.biases[l][j]

                # For each previous node, add its activation multiplied by the corresponding weight
                for k in range(self.dim[l-1]):
                    z += self.activations[l-1][k] * self.weights[l][j][k]

                # Apply activation function
                self.activations[l][j] = sigmoid(z)

        # Return last layer of activations (i.e., the output nodes)
        return self.activations[-1]

    def calc_cost(self, expected):
        ... # Calculate the cost of the network using the MSE cost function

    def train(self, data):
        ... # Train the network given the training data

    def calculate_backprop_gradients(self):
        ... # Calculate how changing each weight and bias in the network
        # affects the overall cost function (backpropagation)

if __name__ == '__main__':
    net = Network(
        [3, 2, 4]
        # Params
    )
    # [Interact with network]
    print(net.calc_network([0.5, 0.1, 0.9]))
