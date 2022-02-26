import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import logging, sys

from utils import perform_statistical_tests
from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

NUM_ITERATIONS = 30

MODELS = [
    GaussianMixture(n_components=2),
    KMeans(n_clusters=3),
    DBSCAN(eps=14, min_samples=130),
    SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0),
    AgglomerativeClustering(n_clusters=3),
]

if __name__ == '__main__':
    df = pd.read_csv('data/census-data.csv')
    encoder = OneHotEncoder()

    features_mutual_info = {feat: pd.DataFrame() for feat in EXTERNAL_FEATURES}

    for _ in range(NUM_ITERATIONS):
        logging.info(f'Running iteration {_}')

        sample_df = df.sample(SAMPLE_SIZE)
        X = encoder.fit_transform(sample_df.drop(EXTERNAL_FEATURES + ['caseid'], axis=1)).toarray()

        y_preds = [model.fit_predict(X) for model in MODELS]

        for feat in EXTERNAL_FEATURES:
            features_mutual_info[feat] = features_mutual_info[feat].append({
                model: mutual_info_score(sample_df[feat], y_pred) for model, y_pred in zip(MODELS, y_preds)
            }, ignore_index=True)

    if len(sys.argv) > 1 and sys.argv[1] == '--save':
        for feat in EXTERNAL_FEATURES:
            features_mutual_info[feat].to_csv(f'results/clusters_{feat}_mutual_info.csv', index=False)

    for feat in EXTERNAL_FEATURES:
        logging.info(f'---- PERFORMING STATISTICAL TESTS FOR {feat} mutual info ----')
        perform_statistical_tests(features_mutual_info[feat], metric=f'{feat} mutual info')