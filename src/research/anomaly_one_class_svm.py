import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
import numpy as np
from prince.mca import MCA
from sklearn.preprocessing import OneHotEncoder
import time

from src.utils import plot_clusters
from src.constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

MODELS = [
    OneClassSVM(kernel='rbf', gamma='scale', nu=0.1),
]

if __name__ == '__main__':
    np.random.seed(int(time.time()))

    df = pd.read_csv('data/census.csv').drop(EXTERNAL_CENSUS_FEATURES + ['caseid'], axis=1)
    enc = OneHotEncoder()

    X = enc.fit_transform(df.sample(5000)).toarray()

    #embedder = TSNE(n_components=2, perplexity=30)
    embedder = MCA(n_components=2)

    plot_clusters(X, MODELS, embedder=embedder)