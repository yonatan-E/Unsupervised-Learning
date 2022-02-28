from tarfile import POSIX_MAGIC
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from prince.mca import MCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import OneClassSVM
import random, time
import seaborn as sns

import matplotlib.pyplot as plt
from utils import plot_clusters
from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

MODELS = [
    DBSCAN(eps=5.5, min_samples=720),
    OneClassSVM(kernel='rbf', gamma='scale', nu=0.1),
]

def plot_model(model, x, points, axe):

    labels = model.fit_predict(x)

    axe.set_axisbelow(True)
    axe.grid(True, linestyle='--')
    axe.set_yticklabels([])
    axe.set_xticklabels([])
    scatter1 = axe.scatter(points[:, 0][labels == -1], points[:, 1][labels == -1], label='Outlier', color='m', s=10, alpha=.4)
    scatter2 = axe.scatter(points[:, 0][labels != -1], points[:, 1][labels != -1], label='Standard', color='c', s=10, alpha=.4)
    legend = axe.legend(loc='upper right', fontsize=14)
    axe.add_artist(legend)
    axe.set_title(f'anomaly detection for {model.__class__.__name__}', fontsize=16)

if __name__ == '__main__':
    np.random.seed(int(time.time()))

    df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)
    enc = OneHotEncoder()
    X = enc.fit_transform(df.sample(5000)).toarray()

    embedder = MCA(n_components=2)
    points = embedder.fit_transform(X).values

    f, axs = plt.subplots(1, 2, figsize=(23, 10))
    f.subplots_adjust(hspace=.3)

    plot_model(MODELS[0], X, points, axs[0])
    plot_model(MODELS[1], X, points, axs[1])
        
    plt.savefig('plots/anomaly_plots.png')
    plt.savefig('plots/anomaly_plots.svg')