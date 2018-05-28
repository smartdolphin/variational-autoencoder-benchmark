from keras.datasets import mnist
import numpy as np


def load_data(target_shape: list):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train, target_shape)
    x_test = np.reshape(x_test, target_shape)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return (x_train, y_train), (x_test, y_test)
