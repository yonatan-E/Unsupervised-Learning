import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder  
import numpy as np
import logging, sys

from utils import perform_statistical_tests, encode_mixed_data
from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

assert len(sys.argv) > 1

dataset = sys.argv[1]
save = len(sys.argv) > 2 and sys.argv[2] == '--save'

NUM_ITERATIONS = 30

if dataset == 'census':
    SAMPLE_SIZE = 20000
    MODELS = [
        GaussianMixture(n_components=3),
        KMeans(n_clusters=4),
        DBSCAN(eps=4.5, min_samples=720),
        SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0),
        AgglomerativeClustering(n_clusters=3),
    ]
elif dataset == 'shoppers':
    SAMPLE_SIZE = 1000
    MODELS = [
        GaussianMixture(n_components=3),
        KMeans(n_clusters=2),
        SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0),
        AgglomerativeClustering(n_clusters=2),
    ]

if __name__ == '__main__':
    if dataset == 'census':
        df = pd.read_csv('data/census.csv').drop(EXTERNAL_CENSUS_FEATURES + ['caseid'], axis=1)
        encoder = OneHotEncoder()
    elif dataset == 'shoppers':
        df = pd.read_csv('data/online-shoppers-intention.csv') \
            .astype(SHOPPERS_DATA_TYPES) \
            .drop(EXTERNAL_SHOPPERS_FEATURES, axis=1)

    silhouette_results_df = pd.DataFrame()
    for _ in range(NUM_ITERATIONS):
        logging.info(f'Running iteration {_}')

        sample_df = df.sample(SAMPLE_SIZE)
        X = encoder.fit_transform(sample_df).toarray() if dataset == 'census' else encode_mixed_data(sample_df)

        silhouette_results_df = silhouette_results_df.append({
            model: silhouette_score(X, model.fit_predict(X)) for model in MODELS
        }, ignore_index=True)

    if save:
        silhouette_results_df.to_csv(f'results/{dataset}/clusters_silhouette.csv', index=False)

    perform_statistical_tests(silhouette_results_df, metric='silhouette')