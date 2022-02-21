import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

NUM_ITERATIONS = 20
BATCH_SIZE = 20000

MODEL = GaussianMixture
PARAM = 'n_components'
VALUES = range(4, 8)

def cluster_and_evaluate(X, models):
    evaluation_results = {'silhouette': {}}

    for model in models:
        y_pred = model.fit_predict(X)

        evaluation_results['silhouette'][model] = silhouette_score(X, y_pred)

    return evaluation_results

if __name__ == '__main__':
    models = [MODEL(**{PARAM: val}) for val in VALUES]

    df = pd.read_csv('data/data.csv').drop('caseid', axis=1)

    silhouette_df = pd.DataFrame()

    for idx in range(NUM_ITERATIONS):
        print(f'Running iteration {idx}')

        X = df.sample(BATCH_SIZE).values
        evaluation_results = cluster_and_evaluate(X, models)

        silhouette_df = silhouette_df.append(evaluation_results['silhouette'], ignore_index=True)

silhouette_scores = silhouette_df.mean().values

plt.plot(VALUES, silhouette_scores, color='blue')
plt.grid(axis='both', alpha=.3)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel(PARAM)
plt.ylabel('Silhouette score')
plt.show()