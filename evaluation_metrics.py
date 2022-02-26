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

from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

NUM_ITERATIONS = 20

<<<<<<< HEAD
MODEL = KMeans
PARAM_NAME = 'n_clusters'
PARAM_VALUES = range(2, 25)
ADDITIONAL_PARAMS = {}
=======
MODEL = DBSCAN
PARAM_NAME = 'eps'
PARAM_VALUES = np.arange(4, 6.1, 0.1)
ADDITIONAL_PARAMS = {'min_samples': 720}
>>>>>>> 1d17aab1006db45512699e5b64628044d98e97e4

if __name__ == '__main__':
    models = [MODEL(**{PARAM_NAME: param}, **ADDITIONAL_PARAMS) for param in PARAM_VALUES]

    df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)
    encoder = OneHotEncoder()

    silhouette_results_df = pd.DataFrame()

    for _ in range(NUM_ITERATIONS):
        logging.info(f'Running iteration {_}')

        X = encoder.fit_transform(df.sample(SAMPLE_SIZE)).toarray()

        silhouette = {}
        for param, model in zip(PARAM_VALUES, models):
            y_pred = model.fit_predict(X)

            labels = np.unique(y_pred)
            if len(labels) == 1 or (len(labels) == 2 and -1 in labels):
                silhouette[param] = 0
            else:
                silhouette[param] = silhouette_score(X[y_pred != -1], y_pred[y_pred != -1])

        silhouette_results_df = silhouette_results_df.append(silhouette, ignore_index=True)

if len(sys.argv) > 1 and sys.argv[1] == '--save':
    silhouette_results_df.to_csv(f'results/{MODEL.__name__}_silhouette.csv', index=False)

silhouette_scores = silhouette_results_df.mean().values

sns.set_theme(style="darkgrid")
sns.lineplot(x=PARAM_VALUES, y=silhouette_scores)
plt.grid(axis='both', alpha=.3)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel(PARAM_NAME)
plt.ylabel('Silhouette score')
plt.title(f'Silhouette score for {MODEL.__name__}')
plt.show()