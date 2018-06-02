from keras.layers import Dropout
from keras.models import Model


def update_dropout_rate(model: Model, dropout_rate=0.0):
    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.rate = dropout_rate

