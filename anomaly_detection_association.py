import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import logging, sys, time

from utils import perform_statistical_tests
from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

np.random.seed(int(time.time()))

NUM_ITERATIONS = 20


if __name__ == '__main__':
    df = pd.read_csv('data/census-data.csv')

    features_mutual_info = {feat: pd.DataFrame() for feat in EXTERNAL_FEATURES}

    for _ in range(NUM_ITERATIONS):
        logging.info(f'Running iteration {_}')

        sample_df = df.sample(SAMPLE_SIZE)
        anomalies_1 = find_anomalies_1(sample_df.drop(EXTERNAL_FEATURES + ['caseid'], axis=1))
        anomalies_2 = find_anomalies_2(sample_df.drop(EXTERNAL_FEATURES + ['caseid'], axis=1))


        for feat in EXTERNAL_FEATURES:
            features_mutual_info[feat] = features_mutual_info[feat].append({
                'method_1': mutual_info_score(sample_df[feat], anomalies_1),
                'method_2': mutual_info_score(sample_df[feat], anomalies_1)
            }, ignore_index=True)

    if len(sys.argv) > 1 and sys.argv[1] == '--save':
        for feat in EXTERNAL_FEATURES:
            features_mutual_info[feat].to_csv(f'results/anomaly_{feat}_mutual_info.csv', index=False)

    for feat in EXTERNAL_FEATURES:
        logging.info(f'---- PERFORMING STATISTICAL TESTS FOR {feat} mutual info ----')
        perform_statistical_tests(features_mutual_info[feat], metric=f'{feat} mutual info')