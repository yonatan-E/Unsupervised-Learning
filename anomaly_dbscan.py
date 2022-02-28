import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from prince.mca import MCA
from sklearn.preprocessing import OneHotEncoder
import random, time

import matplotlib.pyplot as plt
from utils import plot_clusters
from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


if __name__ == '__main__':
    np.random.seed(int(time.time()))

    df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)
    enc = OneHotEncoder()
    X = enc.fit_transform(df.sample(7000)).toarray()

    model = DBSCAN(eps=5.5, min_samples=720)
    labels = model.fit_predict(X)

    embedder = MCA(n_components=2)
    points = embedder.fit_transform(X).values

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.scatter(points[:5000, 0], points[:5000, 1], c=labels[:5000], s=10, alpha=.4)

    plt.savefig('plots/anomaly_dbscan.png')
    plt.savefig('plots/anomaly_dbscan.svg')