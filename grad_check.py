from network import Network
import numpy as np

H = 0.0001

def init_net(x):
    """
    Initialise sample network with:
    - 2 input nodes
    - 3 hidden nodes
    - 3 output nodes

    Pre-set weights and biases of network, and allow one weight
    or bias to be determined by parameter `x`
    """

    net = Network((2, 3, 3))

    # Biases for each layer in the network
    net.biases = [
        None, 
        np.array([0.5, 0.1, -0.2]),
        np.array([0.1, 0.4, 0.1])
    ]

    # Weights for each layer in the network
    net.weights = [
        None,
        np.array([[-0.1, -0.5], [0.5, 0.8], [0.1, 0.3]]),
        np.array([[0.1, -0.5, 0.1], [x, 0.1, -0.5], [0.1, 0.4, 0.1]])
    ]

    return net


def check_grad(x):
    # Calculating the cost before and after incrementation

    net1 = init_net(x)
    net1.calc_network((1, 2))
    c1 = net1.calc_cost((0.3, 0, 0.5))[1]
    print(c1)
    b_grads, w_grads = net1.calculate_backprop_gradients()

    net2 = init_net(x + H)
    net2.calc_network((1, 2))
    c2 = net2.calc_cost((0.3, 0, 0.5))[1]

    m = (c2 - c1) / H
    print(m)

    # # Output all data values of interest
    print(f'Net cost: {c1:.8f}')
    print(f'Gradient: {m:.8f}')
    print()

    print('Bias grads:')
    for bias_grad in b_grads[1:]:
        print(bias_grad)
        print()

    print('Weight grads:')
    for weight_grad in w_grads[1:]:
        print(weight_grad)
        print()

if __name__ == "__main__":
    check_grad(0.5)