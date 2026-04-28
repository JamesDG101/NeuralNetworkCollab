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

    def __init__(self, dimensions, init_random=True):
        self.dim = dimensions

        # Store expected node output values when calculating cost of
        # network, to be used during backprop calcultions
        self.expected = None

        # Empty array for index 0 of weights and biases (as the first layer of the network
        # is just the input nodes -> we don't do anything to these values)
        
        if init_random:
            # Randomised weights and biases
            self.biases  = [np.array([])] + [np.random.randn(j) for j in self.dim[1:]]
            self.weights = [np.array([])] + [np.random.randn(j, k) for j, k in zip(self.dim[1:], self.dim[:-1])]
        else:
            # Weights and biases set to 0
            self.biases  = [np.array([])] + [np.zeros(j) for j in self.dim[1:]]
            self.weights = [np.array([])] + [np.zeros((j, k)) for j, k in zip(self.dim[1:], self.dim[:-1])]

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
        # Calculate the cost of the network using the MSE cost function

        self.expected = expected

        # Calculate differences in network output and expected output, and average
        costs = [(a - y) ** 2 for a, y in zip(self.activations[-1], self.expected)]
        avg_cost = sum(costs) / (len(self.activations) * 2)

        # Determine if the network was correct in its guess
        is_max_matching = np.argmax(self.expected) == np.argmax(self.activations[-1])

        return is_max_matching, avg_cost

    def train(self, data):
        ... # Train the network given the training data

    def calculate_backprop_gradients(self):
        """
        Calculate how changing each weight and bias in the network
        affects the overall cost function (backpropagation)

        IMPORTANT: you must call calc_network AND calc_cost before calling this function    
        """

        # Create new network of same dimensions, where we will calculate & fill
        # each weight and bias with the backpropogation _gradients_
        grad = Network(self.dim, init_random=False)

        # Work backwards through the network's layers
        for l in range(len(self.dim)-1, 0, -1):

            # Calculate activation grads in layer
            for j in range(self.dim[l]):
                if l == len(self.dim)-1:
                    # Final layer: consider the effect of the MSE function on node cost function gradient
                    grad.activations[l][j] += (self.activations[l][j] - self.expected[j]) / self.dim[l]
                else:
                    # Non-final layer: consider the effect of the inputting weights,
                    # the node's bias value and the sigmoid activaion function on cost function gradient
                    k = j # j in current layer is k in next layer
                    for j in range(self.dim[l + 1]):
                        grad.activations[l][k] += grad.activations[l+1][j] * \
                            self.activations[l+1][j] * (1 - self.activations[l+1][j]) * self.weights[l+1][j][k]

            # Calc weight and bias grads leading into current layer from activation gradients
            for j in range(self.dim[l]):

                # Calc bias gradients
                grad.biases[l][j] = grad.activations[l][j] * \
                    self.activations[l][j] * (1 - self.activations[l][j])

                for k in range(self.dim[l-1]):
                    # Calc weight
                    grad.weights[l][j][k] = grad.activations[l][j] * \
                        self.activations[l][j] * (1 - self.activations[l][j]) * self.activations[l-1][k]

        return grad.biases, grad.weights

if __name__ == '__main__':
    net = Network(
        [2, 3, 2]
    )
    print()
    net.calc_network([1, 2])
    print()
    print(net.calc_cost([4, 5]))
    b, w = net.calculate_backprop_gradients()
    print()
    print(b)
    print()
    print(w)
