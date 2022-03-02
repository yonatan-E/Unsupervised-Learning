import pandas as pd
import numpy as np
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder
from prince.mca import MCA
from scipy import interpolate
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import seaborn as sns
import sys, time

assert len(sys.argv) > 1

dataset = sys.argv[1]

from utils import encode_mixed_data
from constants import *

np.random.seed(int(time.time()))

def plot_clustering_association(points, labels, feature, axe):
    for l in np.unique(labels):
        cluster = points[labels==l]
        center = np.mean(cluster, axis=0)

        for point in cluster:
            x = [point[0], center[0]]
            y = [point[1], center[1]]

            axe.plot(x, y, '#e8e2c6', alpha=0.03, linewidth=2)

    axe.set_axisbelow(True)
    axe.grid(True, linestyle='--')
    axe.set_yticklabels([])
    axe.set_xticklabels([])

    scatter = axe.scatter(points[:, 0], points[:, 1], c=feature, s=10, alpha=.4)
    legend = axe.legend(*scatter.legend_elements(), loc='upper right', title='Class')
    axe.add_artist(legend)

if dataset == 'census':
    df = pd.read_csv('data/census.csv')
    sample_df = df.sample(SAMPLE_SIZE)

    X = OneHotEncoder().fit_transform(sample_df.drop(EXTERNAL_CENSUS_FEATURES + ['caseid'], axis=1)).toarray()

    embedder = TSNE(n_components=2, perplexity=60)
    points = embedder.fit_transform(X)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    model = GaussianMixture(n_components=3)
    labels = model.fit_predict(X)

    plot_clustering_association(points, labels, sample_df['dAge'], axs[0])
    axs[0].set_title('K-Means clusters association with dAge external feature')

    plot_clustering_association(points, labels, sample_df['iYearwrk'], axs[1])
    axs[1].set_title('K-Means clusters association with iYearwrk external feature')

    plt.savefig(f'plots/{dataset}/clustering_dAge_iYearwrk_association.png')

if dataset == 'shoppers':
    df = pd.read_csv('data/online-shoppers-intention.csv').astype(SHOPPERS_DATA_TYPES)

    X = encode_mixed_data(df.drop(EXTERNAL_SHOPPERS_FEATURES, axis=1))

    embedder = TSNE(n_components=2, perplexity=30)
    points = embedder.fit_transform(X)

    axe = plt.gca()

    model = GaussianMixture(n_components=3)
    labels = model.fit_predict(X)

    plot_clustering_association(points, labels, df['Revenue'], axe)

    plt.savefig(f'plots/{dataset}/clustering_Revenue_association.png')
    plt.savefig(f'plots/{dataset}/clustering_Revenue_association.svg')