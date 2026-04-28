from network import Network
import numpy as np
import gzip

class Trainer:
    def __init__(self, hidden_nodes, source=None, save_to=None):
        self.net = Network((784, hidden_nodes, 10))

        self.save_to = save_to

        if source is not None:
            try:
                data = np.load(source, allow_pickle=True)
            except:
                print("could not load images")

            self.net.weights = data['weights'].tolist()
            self.net.biases = data['biases'].tolist()
            self.net.size = data['size']

        self.training_data = list(zip(
            self.load_images('data/train-images.gz')/255,
            self.load_labels('data/train-labels.gz')
        ))
        self.test_data = list(zip(
            self.load_images('data/test-images.gz')/255,
            self.load_labels('data/test-labels.gz')
        ))
        print("Data and labels loaded successfully")
    
    @staticmethod
    def load_images(f_name):
        with gzip.open(f_name, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
            images = images.reshape(-1,28*28)
        
        return images
    @staticmethod
    def load_labels(f_name):
        with gzip.open(f_name, 'rb') as f:
            nums = np.frombuffer(f.read(), np.uint8, offset=8)

        
        labels = np.zeros((len(nums),10))
        labels[np.arange(len(nums)), nums] = 1

        return labels

    def do_epoch(self, learning_rate, batch_size):
        pass

    def train_loop(self, learning_rate=3.0, bacth_size=10):
        pass

if __name__ == "__main__":
    tr = Trainer(30)
