import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from prince.mca import MCA

from utils import plot_clusters
from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

MODELS = [
    GaussianMixture(n_components=3),
    KMeans(n_clusters=3),
    KMeans(n_clusters=4),
    KMeans(n_clusters=5),
    DBSCAN(eps=1, min_samples=120),
    DBSCAN(eps=1.3, min_samples=120),
    DBSCAN(eps=1.7, min_samples=120),
    SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0),
    SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0),
    AgglomerativeClustering(n_clusters=3),
    AgglomerativeClustering(n_clusters=4),
    AgglomerativeClustering(n_clusters=5),
]

if __name__ == '__main__':
    df = pd.read_csv('data/reduced-census-data.csv')
    X = df.sample(20000)
    minmax = MinMaxScaler()
    minmax.fit(X)
    X = minmax.transform(X)

    embedder = TSNE(n_components=2, perplexity=50)
    plot_clusters(X, MODELS, embedder=embedder)