import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np

from utils import plot_clusters
from constants import EXTERNAL_FEATURES, SAMPLE_SIZE

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

NUM_CLUSTERS = 9

MODELS = [
    #GaussianMixture(n_components=NUM_CLUSTERS),
    #KMeans(n_clusters=NUM_CLUSTERS),
    DBSCAN(eps=12, min_samples=120),
    #SpectralClustering(n_clusters=NUM_CLUSTERS, n_components=2, affinity='nearest_neighbors'),
    #AgglomerativeClustering(n_clusters=NUM_CLUSTERS),
]

if __name__ == '__main__':
    df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)
    X = df.sample(SAMPLE_SIZE).values

    embedder = TSNE(n_components=2, perplexity=140)
    #embedder = Isomap(n_components=2)
    plot_clusters(X, MODELS, embedder=embedder)