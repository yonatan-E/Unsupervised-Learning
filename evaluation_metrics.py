import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import logging, sys
from prince.mca import MCA
import seaborn as sns

from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

NUM_ITERATIONS = 30

MODEL = GaussianMixture
PARAM_NAME = 'n_components'
PARAM_VALUES = range(2, 26)
ADDITIONAL_PARAMS = {}

if __name__ == '__main__':
    models = [MODEL(**{PARAM_NAME: param}, **ADDITIONAL_PARAMS) for param in PARAM_VALUES]

    df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)
    mca = MCA(n_components=DIMENSIONS, random_state=0)

    silhouette_results_df = pd.DataFrame()

    for _ in range(NUM_ITERATIONS):
        logging.info(f'Running iteration {_}')

        X = mca.fit_transform(df.sample(SAMPLE_SIZE)).values

        silhouette = {}
        for param, model in zip(PARAM_VALUES, models):
            y_pred = model.fit_predict(X)

            silhouette[param] = silhouette_score(X, y_pred)

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