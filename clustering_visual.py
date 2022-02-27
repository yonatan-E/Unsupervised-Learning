import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from prince.mca import MCA
from sklearn.preprocessing import OneHotEncoder

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
    KMeans(n_clusters=6),
    DBSCAN(eps=4.00, min_samples=720),
    DBSCAN(eps=4.20, min_samples=720),
    DBSCAN(eps=4.40, min_samples=720),
    DBSCAN(eps=4.60, min_samples=720),
    DBSCAN(eps=4.80, min_samples=720),
    DBSCAN(eps=5.00, min_samples=720),
    DBSCAN(eps=5.2, min_samples=720),
    DBSCAN(eps=5.4, min_samples=720),
    DBSCAN(eps=5.5, min_samples=720),
    DBSCAN(eps=5.6, min_samples=720),
    DBSCAN(eps=5.8, min_samples=720),
    DBSCAN(eps=6, min_samples=720),
    #SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0),
    #SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0),
    #AgglomerativeClustering(n_clusters=3),
    #AgglomerativeClustering(n_clusters=4),
    #AgglomerativeClustering(n_clusters=5),
]

if __name__ == '__main__':
    df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)
    enc = OneHotEncoder()
    X = enc.fit_transform(df.sample(SAMPLE_SIZE)).toarray()

    embedder = TSNE(n_components=2, perplexity=30)
    #embedder = MCA(n_components=2)
    #embedder = Isomap(n_components=2)
    plot_clusters(X, MODELS, embedder=embedder)