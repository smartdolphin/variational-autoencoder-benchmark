from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def __majority(arr):
    counter = Counter(arr)
    value, _ = counter.most_common(1)[0]
    return value


def clustering_accuracy(y_true, y_clustering):
    clustering_labels = list(set(y_clustering))
    new_labels = np.zeros_like(y_clustering)
    for clustering_label in clustering_labels:
        locator = y_clustering == clustering_label
        locations = np.argwhere(locator)
        real_labels = y_true[locations].ravel()
        major_label = __majority(real_labels)
        new_labels[locator] = major_label
    return accuracy_score(y_true, new_labels)


def confusion_matrix_majority(y_true, y_clustering):
    clustering_labels = list(set(y_clustering))
    new_labels = np.zeros_like(y_clustering)
    for clustering_label in clustering_labels:
        locator = y_clustering == clustering_label
        locations = np.argwhere(locator)
        real_labels = y_true[locations].ravel()
        major_label = __majority(real_labels)
        new_labels[locator] = major_label
    return confusion_matrix(y_true, new_labels)


def visual_cm(z, label, name):
    from sklearn.cluster import KMeans
    N = len(set(label))

    # k-means
    kmeans = KMeans(n_clusters=N, max_iter=2000, n_jobs=1, n_init=20)
    y_pred = kmeans.fit_predict(np.array(z))

    # confusion matrix
    conf_mat = confusion_matrix_majority(np.array(label), y_pred)

    # test metric
    total_count = np.sum(conf_mat)
    correct = np.trace(conf_mat)
    accuracy = correct / total_count
    print('K-means accuracy: {0}'.format(accuracy))