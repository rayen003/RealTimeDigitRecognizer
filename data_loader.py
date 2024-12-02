import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class DigitDataLoader:
    def __init__(self):
        self.digit_data_dir = 'digit_data'

    def download_dataset(self):
        """Download MNIST digit dataset"""
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

    def load_data(self):
        """Load and preprocess digit data"""
        # Normalize and reshape images
        X_train = self.X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        X_test = self.X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

        # Convert labels to categorical
        y_train = to_categorical(self.y_train, 10)
        y_test = to_categorical(self.y_test, 10)

        return X_train, y_train, X_test, y_test
