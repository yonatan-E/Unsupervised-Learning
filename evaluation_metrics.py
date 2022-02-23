import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import logging, sys

from constants import EXTERNAL_FEATURES, SAMPLE_SIZE

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

NUM_ITERATIONS = 30

MODEL = KMeans
PARAM_NAME = 'n_clusters'
PARAM_VALUES = range(2, 16)
# ADDITIONAL_PARAMS = {'affinity': 'nearest_neighbors', 'random_state': 0}
ADDITIONAL_PARAMS = {}

if __name__ == '__main__':
    models = [MODEL(**{PARAM_NAME: param}, **ADDITIONAL_PARAMS) for param in PARAM_VALUES]

    df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)

    silhouette_results_df = pd.DataFrame()

    for _ in range(NUM_ITERATIONS):
        logging.info(f'Running iteration {_}')

        X = df.sample(SAMPLE_SIZE).values

        silhouette = {}
        for param, model in zip(PARAM_VALUES, models):
            y_pred = model.fit_predict(X)

            silhouette[param] = silhouette_score(X, y_pred)

        silhouette_results_df = silhouette_results_df.append(silhouette, ignore_index=True)

if len(sys.argv) > 1 and sys.argv[1] == '--save':
    silhouette_results_df.to_csv(f'results/{MODEL.__name__}_silhouette.csv')

silhouette_scores = silhouette_results_df.mean().values

plt.plot(PARAM_VALUES, silhouette_scores, color='blue')
plt.grid(axis='both', alpha=.3)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel(PARAM_NAME)
plt.ylabel('Silhouette score')
plt.title(f'Silhouette score for {MODEL.__name__}')
plt.show()