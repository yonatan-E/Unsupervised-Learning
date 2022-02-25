import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
from prince.mca import MCA

from utils import plot_clusters
from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

MODELS = [
    GaussianMixture(n_components=20),
    KMeans(n_clusters=20),
    #DBSCAN(eps=13, min_samples=130),
    #SpectralClustering(n_clusters=2, n_components=2, affinity='nearest_neighbors'),
    #AgglomerativeClustering(n_clusters=2),
]

if __name__ == '__main__':
    df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)
    mca = MCA(n_components=DIMENSIONS, random_state=0)
    X = mca.fit_transform(df.sample(SAMPLE_SIZE)).values

    embedder = TSNE(n_components=2, perplexity=50)
    #embedder = Isomap(n_components=2)
    plot_clusters(X, MODELS, embedder=embedder)