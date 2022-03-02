from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from prince.mca import MCA
from prince.famd import FAMD
from sklearn.preprocessing import OneHotEncoder
import sys, time

from utils import plot_clusters, encode_mixed_data
from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

assert len(sys.argv) > 1

dataset = sys.argv[1]

np.random.seed(int(time.time()))

MODELS = [
    GaussianMixture(n_components=3),
    KMeans(n_clusters=2),
    SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0),
    AgglomerativeClustering(n_clusters=2),
    # GaussianMixture(n_components=2),
    # GaussianMixture(n_components=3),
    # KMeans(n_clusters=2),
    # KMeans(n_clusters=3),
    # KMeans(n_clusters=4),
    # KMeans(n_clusters=5),
    # KMeans(n_clusters=6),
    # KMeans(n_clusters=7),
    # KMeans(n_clusters=8),
    # DBSCAN(eps=0.1, min_samples=100),
    # DBSCAN(eps=0.2, min_samples=100),
    # DBSCAN(eps=0.3, min_samples=100),
    # DBSCAN(eps=0.4, min_samples=100),
    # DBSCAN(eps=0.5, min_samples=100),
    # DBSCAN(eps=0.6, min_samples=100),
    # DBSCAN(eps=0.7, min_samples=100),
    # DBSCAN(eps=0.8, min_samples=100),
    # DBSCAN(eps=0.9, min_samples=100),
    # DBSCAN(eps=1, min_samples=100),
    # DBSCAN(eps=1.3, min_samples=100),
    # DBSCAN(eps=1.6, min_samples=100),
    # DBSCAN(eps=1.9, min_samples=100),
    # DBSCAN(eps=2, min_samples=100),
    # DBSCAN(eps=2.1, min_samples=100),
    # DBSCAN(eps=2.3, min_samples=100),
    # DBSCAN(eps=2.5, min_samples=100),
    # SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0),
    # SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0),
    # SpectralClustering(n_clusters=4, affinity='nearest_neighbors', random_state=0),
    # AgglomerativeClustering(n_clusters=2),
    # AgglomerativeClustering(n_clusters=3),
    # AgglomerativeClustering(n_clusters=4),
    # AgglomerativeClustering(n_clusters=5),
]

if __name__ == '__main__':
    if dataset == 'census':
        df = pd.read_csv('data/census.csv').drop(EXTERNAL_CENSUS_FEATURES + ['caseid'], axis=1)
        X = OneHotEncoder().fit_transform(df.sample(SAMPLE_SIZE)).toarray()
    elif dataset == 'shoppers':
        df = pd.read_csv('data/online-shoppers-intention.csv') \
            .astype(SHOPPERS_DATA_TYPES) \
            .drop(EXTERNAL_SHOPPERS_FEATURES, axis=1)
        X = encode_mixed_data(df)

    plot_clusters(X, MODELS, embedder=TSNE(n_components=2))