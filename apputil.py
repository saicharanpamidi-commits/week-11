from time import time

import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans


def kmeans(X, k):
    """
    Run k-means clustering on a numeric NumPy array X.

    Returns a tuple (centroids, labels):
    - centroids has shape (k, n_features), one row per cluster
    - labels has shape (n_samples,), the cluster index for each row of X
    """
    model = KMeans(n_clusters=k)
    model.fit(np.asarray(X))

    centroids = model.cluster_centers_
    labels = model.labels_

    return centroids, labels


# Exercise 2: load the diamonds data once and keep only the numeric columns
# (carat, depth, table, price, x, y, z)
diamonds = sns.load_dataset("diamonds")
diamonds_numeric = diamonds.select_dtypes(include="number")


def kmeans_diamonds(n, k):
    """Run kmeans with k clusters on the first n rows of numeric diamonds."""
    X = diamonds_numeric.iloc[:n].to_numpy()
    return kmeans(X, k)


def kmeans_timer(n, k, n_iter=5):
    """
    Run kmeans_diamonds(n, k) n_iter times and return the average runtime
    in seconds.
    """
    times = []

    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        times.append(time() - start)

    return sum(times) / len(times)
