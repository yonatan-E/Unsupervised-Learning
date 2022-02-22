import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import logging, sys

from utils import calculate_dunn_index, plot_clusters, perform_statistical_tests
from constants import EXTERNAL_FEATURES, SAMPLE_SIZE

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

NUM_CLUSTERS = 9
NUM_ITERATIONS = 20

MODELS = [
    GaussianMixture(n_components=NUM_CLUSTERS),
    KMeans(n_clusters=NUM_CLUSTERS),
    DBSCAN(eps=100, min_samples=8),
    #SpectralClustering(n_clusters=NUM_CLUSTERS, n_components=2, affinity='nearest_neighbors'),
    AgglomerativeClustering(n_clusters=NUM_CLUSTERS),
]

def cluster_and_evaluate(X):
    evaluation_results = {'silhouette': {}, 'dunn_index': {}}

    for model in MODELS:
        y_pred = model.fit_predict(X)

        evaluation_results['silhouette'][model] = silhouette_score(X, y_pred)
        evaluation_results['dunn_index'][model] = calculate_dunn_index(X, y_pred)

    return evaluation_results

if __name__ == '__main__':
    df = pd.read_csv('data/original-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)

    silhouette_results_df = pd.DataFrame()
    dunn_results_df = pd.DataFrame()

    for _ in range(NUM_ITERATIONS):
        print(f'Running iteration {_}')

        X = df.sample(SAMPLE_SIZE).values
        evaluation_results = cluster_and_evaluate(X)

        silhouette_results_df = silhouette_results_df.append(evaluation_results['silhouette'])
        dunn_results_df = dunn_results_df.append(evaluation_results['dunn'])

    if len(sys.argv) > 1 and sys.argv[1] == '--save':
        silhouette_results_df.to_csv('results/clusters_silhouette.csv')
        dunn_results_df.to_csv('results/clusters_dunn_index.csv')

    perform_statistical_tests(silhouette_results_df, metric='silhouette')
    perform_statistical_tests(dunn_results_df, metric='dunn index')