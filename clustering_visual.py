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
import time, sys

from utils import plot_clusters, encode_mixed_data
from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

MODELS = [
    GaussianMixture(n_components=10),
    KMeans(n_clusters=10),
    #SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0),
    #SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0),
    #AgglomerativeClustering(n_clusters=3),
    #AgglomerativeClustering(n_clusters=4),
    #AgglomerativeClustering(n_clusters=5),
]

if __name__ == '__main__':
    np.random.seed(int(time.time()))

    if sys.argv[1] == 'census':
        df = pd.read_csv('data/census.csv').drop(EXTERNAL_CENSUS_FEATURES + ['caseid'], axis=1)
        X = OneHotEncoder().fit_transform(df.sample(SAMPLE_SIZE)).toarray()
    elif sys.argv[1] == 'shoppers':
        df = pd.read_csv('data/online-shoppers-intention.csv') \
            .astype(SHOPPERS_DATA_TYPES) \
            .drop(EXTERNAL_SHOPPERS_FEATURES, axis=1)
        X = encode_mixed_data(df)

    labels = KMeans(n_clusters=5).fit_predict(X)
    #embedder = FAMD(n_components=2)
    embedder = TSNE(n_components=2, perplexity=30)
    points = embedder.fit_transform(X)

    plt.grid(True, linestyle='--')
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, s=10, alpha=.4)
    plt.show()

    #embedder = FAMD(n_components=2)
    #embedder = Isomap(n_components=2)
    #plot_clusters(X, MODELS, embedder=embedder)