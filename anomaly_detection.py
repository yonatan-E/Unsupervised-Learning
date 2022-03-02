import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from prince.mca import MCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
import sys, time

assert len(sys.argv) > 1

dataset = sys.argv[1]

from utils import encode_mixed_data
from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

np.random.seed(int(time.time()))

if dataset == 'census':
    EXTERNAL_FEATURES = EXTERNAL_CENSUS_FEATURES
    MODELS = [
        DBSCAN(eps=5.5, min_samples=720),
        OneClassSVM(kernel='rbf', gamma='scale', nu=0.1),
    ]
elif dataset == 'shoppers':
    EXTERNAL_FEATURES = EXTERNAL_SHOPPERS_FEATURES
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
    if dataset == 'census':
        df = pd.read_csv('data/census.csv').drop(EXTERNAL_CENSUS_FEATURES + ['caseid'], axis=1)
        X = OneHotEncoder().fit_transform(df.sample(5000)).toarray()

        embedder = MCA(n_components=2)
        points = embedder.fit_transform(X).values
    elif dataset == 'shoppers':
        df = pd.read_csv('data/online-shoppers-intention.csv') \
            .astype(SHOPPERS_DATA_TYPES) \
            .drop(EXTERNAL_SHOPPERS_FEATURES, axis=1)
        X = encode_mixed_data(df)

        embedder = TSNE(n_components=2)
        points = embedder.fit_transform(X)

    f, axs = plt.subplots(1, 2, figsize=(23, 10))
    f.subplots_adjust(hspace=.3)

    plot_model(MODELS[0], X, points, axs[0])
    plot_model(MODELS[1], X, points, axs[1])
        
    plt.savefig(f'plots/{dataset}/anomaly_plots.png')
    plt.savefig(f'plots/{dataset}/anomaly_plots.svg')