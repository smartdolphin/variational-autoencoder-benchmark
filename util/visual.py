from sklearn.cluster import KMeans
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from util.metric import clustering_accuracy, confusion_matrix_majority


def kmeans_clustering_accuracy(z, y_true):
    N = len(set(y_true))

    # k-means
    kmeans = KMeans(n_clusters=N, max_iter=2000, n_jobs=1, n_init=20)
    y_pred = kmeans.fit_predict(np.array(z))

    # confusion matrix
    accuracy = clustering_accuracy(np.array(y_true), y_pred)
    print('K-means accuracy: {0}'.format(accuracy))


def kmeans_confusion_matrix(latent_z, y_true, save_path, class_names=None):
    N = len(set(y_true))

    # k-means
    kmeans = KMeans(n_clusters=N, max_iter=2000, n_jobs=1, n_init=20)
    y_pred = kmeans.fit_predict(np.array(latent_z))

    # confusion matrix
    conf_mat = confusion_matrix_majority(np.array(y_true), y_pred)

    # test metric
    total_count = np.sum(conf_mat)
    correct = np.trace(conf_mat)
    accuracy = correct / total_count
    title = 'K-means accuracy: {0}'.format(accuracy)
    print(title)
    plot_confusion_matrix(conf_mat, class_names, save_path, False, title)


def plot_confusion_matrix(cm, class_names, save_path, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arrange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.clf()


# referenced by keras team(https://github.com/keras-team)
def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()
