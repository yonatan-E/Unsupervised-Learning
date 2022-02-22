import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score
import numpy as np
import logging, sys

from utils import plot_clusters, perform_statistical_tests
from constants import EXTERNAL_FEATURES, SAMPLE_SIZE

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

NUM_CLUSTERS = 9
NUM_ITERATIONS = 20

FEATURE = 'dAge'

MODELS = [
    GaussianMixture(n_components=NUM_CLUSTERS),
    KMeans(n_clusters=NUM_CLUSTERS),
    DBSCAN(eps=100, min_samples=8),
    #SpectralClustering(n_clusters=NUM_CLUSTERS, n_components=2, affinity='nearest_neighbors'),
    AgglomerativeClustering(n_clusters=NUM_CLUSTERS),
]

if __name__ == '__main__':
    df = pd.read_csv('data/original-data.csv')

    mutual_info_results_df = pd.DataFrame()

    for _ in range(NUM_ITERATIONS):
        logging.info(f'Running iteration {_}')

        sample_df = df.drop(EXTERNAL_FEATURES + ['caseid'], axis=1)
        X = sample_df.sample(SAMPLE_SIZE).values
        external_feature = sample_df[FEATURE]

        mutual_info_results_df = mutual_info_results_df.append({
            model: mutual_info_score(external_feature, model.fit_predict(X)) for model in MODELS
        }, ignore_index=True)

    if len(sys.argv) > 1 and sys.argv[1] == '--save':
        mutual_info_results_df.to_csv(f'results/clusters_{FEATURE}_mutual_info_.csv')

    perform_statistical_tests(mutual_info_results_df, metric='mutual info')