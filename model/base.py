from abc import abstractmethod, ABC
from sklearn.metrics import log_loss, mean_squared_error

class Encoder(ABC):
    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, encoded_x):
        pass

    def recon_error(self, x, metric='cross_entropy'):
        encoded_x = self.encode(x)
        decoded_x = self.decode(encoded_x)
        if metric == 'cross_entropy':
            error = log_loss(x, decoded_x)
        elif metric == 'mean_square_error':
            error = mean_squared_error(x, decoded_x)
        else:
            raise ValueError('%s metric is not supported' % metric)
        return error