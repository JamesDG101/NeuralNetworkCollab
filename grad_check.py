from network import Network
import numpy as np

H = 0.00001

def do_check(v):
    # Calculating the cost before and after incrementation
    c1, grads = calc_cost(v)
    c2 = calc_cost(v + H)[0]

    m = (c2-c1)/H

    # Output all data values of interest
    print(f'Net cost: {c1:.8f}')
    print(f'Gradient: {m:.8f}')
    print()

    bias_grads, weight_grads = grads

    print('Bias grads:')
    for bias_grad in bias_grads[1:]:
        print(bias_grad)
        print()

    print('Weight grads:')
    for weight_grad in weight_grads[1:]:
        print(weight_grad)
        print()


def calc_cost(x):
    net = Network((2,3,3)) # 2 inputs 3 hidden nodes and 3 outputs 

    net.biases = [None, 
                  np.array([0.5,0.1,-0.2]),
                  np.array([0.1,0.4,x])] # Here are the biases for each layer in the network
    net.weights = [None,
                   np.array([[-0.1,-0.5],[0.5,0.8],[0.1,0.3]]),
                   np.array([[0.1,-0.5,0.1],[0.6,0.1,-0.5],[0.1,0.4,0.1]])] # Same for the weights

    return net.calc_network((1,2),(0.5,-0.2,0.2))

if __name__ == "__main__":
    do_check(0.5)