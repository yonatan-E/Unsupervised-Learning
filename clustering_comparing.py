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

NUM_ITERATIONS = 30

MODELS = [
    GaussianMixture(n_components=2),
    KMeans(n_clusters=3),
    DBSCAN(eps=13, min_samples=120),
    SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0),
    AgglomerativeClustering(n_clusters=3),
]

if __name__ == '__main__':
    df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)

    silhouette_results_df = pd.DataFrame()

    for _ in range(NUM_ITERATIONS):
        logging.info(f'Running iteration {_}')

        X = df.sample(SAMPLE_SIZE).values

        silhouette = {}
        for model in MODELS:
            y_pred = model.fit_predict(X)

            silhouette[model] = silhouette_score(X, y_pred)

        silhouette_results_df = silhouette_results_df.append(silhouette, ignore_index=True)

    if len(sys.argv) > 1 and sys.argv[1] == '--save':
        silhouette_results_df.to_csv('results/clusters_silhouette.csv', index=False)

    perform_statistical_tests(silhouette_results_df, metric='silhouette')