import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import logging, sys, time
from sklearn.svm import OneClassSVM

from src.utils import perform_statistical_tests
from src.constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

np.random.seed(int(time.time()))

NUM_ITERATIONS = 20

assert len(sys.argv) > 1

dataset = sys.argv[1]
save = len(sys.argv) > 2 and sys.argv[2] == '--save'

if dataset == 'census':
    SAMPLE_SIZE = 20000
    EXTERNAL_FEATURES = EXTERNAL_CENSUS_FEATURES
    MODELS = [
    DBSCAN(eps=5.5, min_samples=720),
    OneClassSVM(kernel='rbf', gamma='scale', nu=0.1),
    ]

elif dataset == 'shoppers':
    SAMPLE_SIZE = 1000
    EXTERNAL_FEATURES = EXTERNAL_SHOPPERS_FEATURES
    MODELS = [
    DBSCAN(eps=5.5, min_samples=720),
    OneClassSVM(kernel='rbf', gamma='scale', nu=0.1),
    ]   


if __name__ == '__main__':

    df = pd.read_csv('data/census-data.csv')

    features_mutual_info = {feat: pd.DataFrame() for feat in EXTERNAL_FEATURES}

    enc = OneHotEncoder()

    for _ in range(NUM_ITERATIONS):
        logging.info(f'Running iteration {_}')

        sample_df = df.sample(SAMPLE_SIZE)
        X = enc.fit_transform(sample_df.drop(EXTERNAL_FEATURES + ['caseid'], axis=1)).toarray()
        anomalies_1 = [1 if l == -1 else 0 for l in MODELS[0].fit_predict(X)]
        anomalies_2 = [1 if l ==-1 else 0 for l in MODELS[1].fit_predict(X)] 
        for feat in EXTERNAL_FEATURES:
            features_mutual_info[feat] = features_mutual_info[feat].append({
                'DBSCAN': mutual_info_score(sample_df[feat], anomalies_1),
                'OneClassSVM': mutual_info_score(sample_df[feat], anomalies_2)
            }, ignore_index=True)

    if len(sys.argv) > 1 and sys.argv[1] == '--save':
        for feat in EXTERNAL_FEATURES:
            features_mutual_info[feat].to_csv(f'results/{dataset}/anomaly_{feat}_mutual_info.csv', index=False)

    for feat in EXTERNAL_FEATURES:
        logging.info(f'---- PERFORMING STATISTICAL TESTS FOR {feat} mutual info ----')
        perform_statistical_tests(features_mutual_info[feat], metric=f'{feat} mutual info')