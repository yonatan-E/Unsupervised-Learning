import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import logging, sys
import seaborn as sns

from utils import encode_mixed_data
from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

assert len(sys.argv) > 1

dataset = sys.argv[1]
save = len(sys.argv) > 2 and sys.argv[2] == '--save'

MODEL = KMeans
PARAM_NAME = 'n_clusters'
PARAM_VALUES = range(2, 16)
ADDITIONAL_PARAMS = {}

def evaluate_models_silhouette(X, models):
    silhouette = {}
    for param, model in zip(PARAM_VALUES, models):
        y_pred = model.fit_predict(X)

        silhouette[param] = silhouette_score(X, y_pred)

    return silhouette

if __name__ == '__main__':
    models = [MODEL(**{PARAM_NAME: param}, **ADDITIONAL_PARAMS) for param in PARAM_VALUES]

    if dataset == 'census':
        df = pd.read_csv('data/census.csv').drop(EXTERNAL_CENSUS_FEATURES + ['caseid'], axis=1)
        encoder = OneHotEncoder()

        silhouette_results_df = pd.DataFrame()
        for _ in range(20):
            logging.info(f'Running iteration {_}')

            X = encoder.fit_transform(df.sample(SAMPLE_SIZE)).toarray()
            silhouette_results_df = silhouette_results_df.append(
                evaluate_models_silhouette(X, models),
                ignore_index=True
            )

    elif dataset == 'shoppers':
        df = pd.read_csv('data/online-shoppers-intention.csv') \
            .astype(SHOPPERS_DATA_TYPES) \
            .drop(EXTERNAL_SHOPPERS_FEATURES, axis=1)
        X = encode_mixed_data(df)

        silhouette_results_df = pd.DataFrame([evaluate_models_silhouette(X, models)])

    if save:
        silhouette_results_df.to_csv(f'results/{dataset}/{MODEL.__name__}_silhouette.csv', index=False)

    silhouette_scores = silhouette_results_df.mean().values

    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.lineplot(x=PARAM_VALUES, y=silhouette_scores)
    plt.xticks(fontsize=7, alpha=.7)
    plt.yticks(fontsize=7, alpha=.7)
    plt.xlabel(PARAM_NAME)
    plt.ylabel('Silhouette score')
    plt.title(f'Silhouette score for {MODEL.__name__}')
    plt.show()