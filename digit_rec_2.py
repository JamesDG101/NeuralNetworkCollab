import gzip
import numpy as np
from network import Network

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
        images = images.reshape(-1, 28*28)
    return images

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

train_images = load_images('data/train-images.gz')
train_labels = load_labels('data/train-labels.gz')

test_images = load_images('data/test-images.gz')
test_labels = load_labels('data/test-labels.gz')

class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)

        # biases[layer # -1][node #]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # weights[layer end # -1][end node #][start node #]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        return a
    
    def SGD(self, training_data, epochs):
        pass

    def update_mini_batch(self):
        pass

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

if __name__ == "__main__":
    net = Network([784, 30, 10])
