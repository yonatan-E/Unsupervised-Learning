import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind, f_oneway
import numpy as np
from itertools import combinations, product
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

NUM_CLUSTERS = 9
SIGNIFICANCE_LEVEL = 0.99
NUM_ITERATIONS = 20
BATCH_SIZE = 200

MODELS = [
    GaussianMixture(n_components=NUM_CLUSTERS),
    KMeans(n_clusters=NUM_CLUSTERS),
    DBSCAN(eps=100, min_samples=8),
    SpectralClustering(n_clusters=NUM_CLUSTERS, n_components=2, affinity='nearest_neighbors'),
    AgglomerativeClustering(n_clusters=NUM_CLUSTERS),
]

if __name__ == '__main__':
    df = pd.read_csv('data/data.csv', nrows=1000).drop('caseid', axis=1)

    X = df.values

    X_transformed = TSNE(n_components=2, perplexity=20).fit_transform(X)

    n = np.ceil(np.sqrt(len(MODELS) + 1)).astype(int)

    _, axs = plt.subplots(n, n)

    for idx, model in enumerate(MODELS):

        axs[int(idx / n), idx % n].scatter([x[0] for x in X_transformed], [x[1] for x in X_transformed], c=model.fit_predict(X), s=10)
        axs[int(idx / n), idx % n].set_title(model.__class__.__name__)

    for ax in axs.flat:
        ax.label_outer()

    plt.show()
