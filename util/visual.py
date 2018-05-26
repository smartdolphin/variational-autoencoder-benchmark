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
