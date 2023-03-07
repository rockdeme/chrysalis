from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def silhouette_method(X, max_clusters):
    # Calculate silhouette scores for each number of clusters
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Find the optimal number of clusters with the highest silhouette score
    optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

    return optimal_n_clusters


def elbow_method(X, max_clusters):
    # Calculate sum of squared distances for each number of clusters
    sse = []
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.plot(range(1, max_clusters + 1), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.show()

    # Find the optimal number of clusters based on the elbow plot
    optimal_n_clusters = 1
    min_diff = float('inf')
    for i in range(1, max_clusters):
        diff = sse[i] - sse[i - 1]
        if diff < min_diff:
            min_diff = diff
            optimal_n_clusters = i + 1

    return optimal_n_clusters


def gap_statistic(X, max_clusters):
    # Calculate within-cluster dispersion for each number of clusters
    def calculate_wk(data, labels, centers):
        k = centers.shape[0]
        return sum([np.linalg.norm(data[labels == i] - centers[i]) ** 2 / (2 * sum(labels == i)) for i in range(k)])

    # Generate reference datasets
    def generate_reference_data(X):
        reference_data = np.random.rand(*X.shape)
        return reference_data

    # Calculate gap statistic for each number of clusters
    def calculate_gap_statistic(X, max_clusters):
        gaps = np.zeros((max_clusters,))
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            wk = calculate_wk(X, kmeans.labels_, kmeans.cluster_centers_)
            reference_wks = []
            for i in range(10):
                reference_data = generate_reference_data(X)
                reference_kmeans = KMeans(n_clusters=k)
                reference_kmeans.fit(reference_data)
                reference_wk = calculate_wk(reference_data, reference_kmeans.labels_, reference_kmeans.cluster_centers_)
                reference_wks.append(reference_wk)
            gaps[k - 1] = np.mean(np.log(reference_wks)) - np.log(wk)
        return gaps

    # Calculate gap statistic and find the optimal number of clusters
    gap_values = calculate_gap_statistic(X, max_clusters)
    optimal_n_clusters = np.argmax(gap_values) + 1

    return optimal_n_clusters
