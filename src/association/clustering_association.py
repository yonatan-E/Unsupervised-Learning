import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import logging, sys, time

from src.utils import perform_statistical_tests, encode_mixed_data
from src.constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

np.random.seed(int(time.time()))

assert len(sys.argv) > 1

dataset = sys.argv[1]
save = len(sys.argv) > 2 and sys.argv[2] == '--save'

NUM_ITERATIONS = 30

if dataset == 'census':
    SAMPLE_SIZE = 20000
    EXTERNAL_FEATURES = EXTERNAL_CENSUS_FEATURES
    MODELS = [
        GaussianMixture(n_components=3),
        KMeans(n_clusters=4),
        DBSCAN(eps=4.5, min_samples=720),
        SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0),
        AgglomerativeClustering(n_clusters=3),
    ]
elif dataset == 'shoppers':
    SAMPLE_SIZE = 1000
    EXTERNAL_FEATURES = EXTERNAL_SHOPPERS_FEATURES
    MODELS = [
       GaussianMixture(n_components=3),
       KMeans(n_clusters=2),
       SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0),
       AgglomerativeClustering(n_clusters=2),
    ]

if __name__ == '__main__':
    if dataset == 'census':
        df = pd.read_csv('data/census.csv')
        encoder = OneHotEncoder()
    elif dataset == 'shoppers':
        df = pd.read_csv('data/online-shoppers-intention.csv').astype(SHOPPERS_DATA_TYPES)

    features_mutual_info = {feat: pd.DataFrame() for feat in EXTERNAL_FEATURES}

    for _ in range(NUM_ITERATIONS):
        logging.info(f'Running iteration {_}')

        sample_df = df.sample(SAMPLE_SIZE)
        if dataset == 'census':
            X = encoder.fit_transform(sample_df.drop(EXTERNAL_FEATURES + ['caseid'], axis=1)).toarray()
        elif dataset == 'shoppers':
            X = encode_mixed_data(sample_df.drop(EXTERNAL_FEATURES, axis=1))

        y_preds = [model.fit_predict(X) for model in MODELS]

        for feat in EXTERNAL_FEATURES:
            features_mutual_info[feat] = features_mutual_info[feat].append({
                model: mutual_info_score(sample_df[feat][y_pred != -1], y_pred[y_pred != -1]) for model, y_pred in zip(MODELS, y_preds)
            }, ignore_index=True)

    if save:
        for feat in EXTERNAL_FEATURES:
            features_mutual_info[feat].to_csv(f'results/{dataset}/clusters_{feat}_mutual_info.csv', index=False)

    for feat in EXTERNAL_FEATURES:
        logging.info(f'---- PERFORMING STATISTICAL TESTS FOR {feat} mutual info ----')
        perform_statistical_tests(features_mutual_info[feat], metric=f'{feat} mutual info')