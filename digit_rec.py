import numpy as np
import gzip
import random
import sys

from network import Network

class Trainer:
    def __init__(self, hidden_nodes, source=None, save_to=None):
        self.net = Network((784, hidden_nodes, 10))

        self.save_to = save_to

        if source is not None:
            data = np.load(source, allow_pickle=True)

            self.net.weights = data['weights'].tolist()
            self.net.biases = data['biases'].tolist()
            self.net.size = data['size']

        self.training_data = list(zip(
            self.load_images('data/train-images.gz') / 255,
            self.load_labels('data/train-labels.gz')
        ))
        self.test_data = list(zip(
            self.load_images('data/test-images.gz') / 255,
            self.load_labels('data/test-labels.gz')
        ))
    
    @staticmethod
    def load_images(f_name):
        with gzip.open(f_name, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
            images = images.reshape(-1, 28 * 28)
        
        return images

    @staticmethod
    def load_labels(f_name):
        with gzip.open(f_name, 'rb') as f:
            nums = np.frombuffer(f.read(), np.uint8, offset=8)

        labels = np.zeros((len(nums), 10))
        labels[np.arange(len(nums)), nums] = 1

        return labels

    def do_epoch(self, learning_rate, batch_size):
        past = []
        PAST_LEN = 50

        # Create re-randomised batches of training data
        random.shuffle(self.training_data)
        batches = [
            self.training_data[k:k+batch_size]
            for k in range(0, len(self.training_data), batch_size)
        ]

        for batch_idx, batch in enumerate(batches):
            num_correct, avg_cost = self.net.train_batch(batch, learning_rate)

            sys.stdout.write('\033c')
            sys.stdout.flush()

            perc = num_correct / batch_size
            print(f'Batch {batch_idx + 1} of {len(batches)}:\t{num_correct:>2}/{batch_size}={perc*100:>2.0f}% (Avg cost: {avg_cost:.4f}). W&Bs Updated.')

            past.insert(0, (perc, avg_cost))
            past = past[:PAST_LEN]

            mean_perc = sum([elem[0] for elem in past]) / len(past)
            mean_cost = sum([elem[1] for elem in past]) / len(past)

            print(f'{mean_perc*100:>3.2f}%  {mean_cost:.5f}')

    def train_loop(self, learning_rate=3.0, batch_size=10):
        try:
            while True:
                self.do_epoch(learning_rate, batch_size)

                print('Completed epoch')
        except KeyboardInterrupt:
            print('\n\n')

            if self.save_to is not None:
                np.savez(
                    self.save_to,
                    weights=np.array(self.net.weights, dtype=object),
                    biases=np.array(self.net.biases, dtype=object),
                    size=self.net.size,
                )
                print(f'Saved to \'{self.save_to}\'')

                # print('\n\n')
                # print(self.net.weights)
                # print(self.net.biases)

if __name__ == "__main__":
    tr = Trainer(30)
    tr.train_loop(
        learning_rate=3.0,
        batch_size=10
    )
