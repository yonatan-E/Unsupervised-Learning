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

if __name__ == '__main__':
    if dataset == 'census':
        MODELS = [
            GaussianMixture(n_components=3),
            KMeans(n_clusters=4),
            DBSCAN(eps=4.5, min_samples=720),
            SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0),
            AgglomerativeClustering(n_clusters=3),
        ]

        df = pd.read_csv('data/census.csv').drop(EXTERNAL_CENSUS_FEATURES + ['caseid'], axis=1)
        X = OneHotEncoder().fit_transform(df.sample(SAMPLE_SIZE)).toarray()
    elif dataset == 'shoppers':
        MODELS = [
            GaussianMixture(n_components=3),
            KMeans(n_clusters=2),
            SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0),
            AgglomerativeClustering(n_clusters=2),
        ]

        df = pd.read_csv('data/online-shoppers-intention.csv') \
            .astype(SHOPPERS_DATA_TYPES) \
            .drop(EXTERNAL_SHOPPERS_FEATURES, axis=1)
        X = encode_mixed_data(df)

    plot_clusters(X, MODELS, embedder=TSNE(n_components=2))