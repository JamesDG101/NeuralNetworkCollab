import time
import random
import gzip
import numpy as np

IMG_SIZE = 28 * 28

class Network:
    def __init__(self, sizes, training_data_dir, training_labels_dir,
                 test_data_dir, test_labels_dir):
        
        self.sizes = sizes
        self.num_layers = len(sizes)

        # biases[layer # -1][node #]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # weights[layer end # -1][end node #][start node #]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.training_data = self.load_MNIST(training_data_dir, training_labels_dir)
        self.test_data = self.load_MNIST(test_data_dir, test_labels_dir)

    @staticmethod
    def load_MNIST(data_dir, labels_dir):

        with gzip.open(data_dir, 'rb') as f:
            tests = np.frombuffer(f.read(), np.uint8, offset=16) / 255
            tests = tests.reshape(-1, IMG_SIZE, 1)

        with gzip.open(labels_dir, 'rb') as f:
            nums = np.frombuffer(f.read(), np.uint8, offset=8)

            labels = np.zeros((len(nums), 10))
            labels[np.arange(len(nums)), nums] = 1

        return list(zip(tests, labels))
    

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    

    @staticmethod
    def sigmoid_prime(z):
        return Network.sigmoid(z) * (1 - Network.sigmoid(z))
    

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(w @ a + b)
        return a
    

    def SGD(self, batch_size, learning_rate):
        try:
            epoch_number = 0
            while True:
                epoch_number += 1
                print(f'\nStarting epoch #{epoch_number}...')
                start_time = time.time()

                idxs = list(range(len(self.training_data)))
                random.shuffle(idxs)

                for batch_start in range(0, len(self.training_data), batch_size):
                    batch_idxs = idxs[batch_start:batch_start + batch_size]

                    self.train_batch(batch_idxs, learning_rate)
                
                print('Training complete. Running tests...')

                score = self.evaluate()
                print(f'Score: {score*100:.2f}%')

                time_diff = time.time() - start_time
                mins, secs = divmod(time_diff, 60)
                secs = str(int(secs)).rjust(2, '0')
                print(f'Epoch duration: {int(mins)}:{secs}')

        except KeyboardInterrupt:
            print('\nTraining interrupted.')
        

    def train_batch(self, batch_idxs, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for batch_idx in batch_idxs:
            test, label = self.training_data[batch_idx]
            delta_nabla_b, delta_nabla_w = self.backprop(test, label)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        
        self.weights = [w - (learning_rate * nw / len(batch_idxs))
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate * nb / len(batch_idxs))
                       for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, test, label):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        a = test
        a_vals = [a]  # Network activations
        z_vals = []  # Network z values

        for b, w in zip(self.biases, self.weights):
            z = w @ a + b
            z_vals.append(z)
            a = self.sigmoid(z)
            a_vals.append(a)
            
        delta = (a_vals[-1] - label.reshape(-1, 1)) * self.sigmoid_prime(z_vals[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = delta @ a_vals[-2].T

        for l in range(2, self.num_layers):
            z = z_vals[-l]
            sp = self.sigmoid_prime(z)
            delta = self.weights[-l + 1].T @ delta * sp

            nabla_b[-l] = delta
            nabla_w[-l] = delta @ a_vals[-l - 1].T
            
        return nabla_b, nabla_w
        
    def evaluate(self):
        return sum([np.argmax(self.feedforward(test)) == np.argmax(label) 
                    for test, label in self.test_data]) / len(self.test_data)
    
if __name__ == '__main__':
    net = Network(
        sizes=[IMG_SIZE, 15, 10],
        training_data_dir='data/train-images.gz',
        training_labels_dir='data/train-labels.gz',
        test_data_dir='data/test-images.gz',
        test_labels_dir='data/test-labels.gz'
    )

    net.SGD(15, 3.0)
