import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

from utils import calculate_dunn_index
from constants import SAMPLE_SIZE

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

NUM_ITERATIONS = 30

MODEL = GaussianMixture
PARAM_NAME = 'n_components'
PARAM_VALUES = range(2, 16)

if __name__ == '__main__':
    models = [MODEL(**{PARAM_NAME: param}) for param in PARAM_VALUES]

    df = pd.read_csv('data/data.csv').drop('caseid', axis=1)

    silhouette_results_df = pd.DataFrame()
    dunn_index_results_df = pd.DataFrame()

    for _ in range(NUM_ITERATIONS):
        print(f'Running iteration {_}')

        X = df.sample(SAMPLE_SIZE).values

        silhouette, dunn_index = {}, {}
        for param, model in zip(PARAM_VALUES, models):
            y_pred = model.fit_predict(X)

            silhouette[param] = silhouette_score(X, y_pred)
            dunn_index[param] = calculate_dunn_index(X, y_pred)

        silhouette_results_df = silhouette_results_df.append(silhouette, ignore_index=True)
        dunn_index_results_df = dunn_index_results_df.append(dunn_index, ignore_index=True)

        break

silhouette_results_df.to_csv(f'{MODEL.__name__}_silhouette.csv')
dunn_index_results_df.to_csv(f'{MODEL.__name__}_dunn_index.csv')

exit()

silhouette_scores = silhouette_results_df.mean().values
dunn_index_scores = dunn_index_results_df.mean().values

plt.plot(PARAM_VALUES, silhouette_scores, color='blue')
plt.grid(axis='both', alpha=.3)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel(PARAM_NAME)
plt.ylabel('Silhouette score')
plt.show()