import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind, f_oneway
import numpy as np
from itertools import combinations, product
import matplotlib.pyplot as plt

from utils import plot_clusters

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

NUM_CLUSTERS = 9
SIGNIFICANCE_LEVEL = 0.99
NUM_ITERATIONS = 20
BATCH_SIZE = 200

MODELS = [
    #GaussianMixture(n_components=NUM_CLUSTERS),
    #KMeans(n_clusters=NUM_CLUSTERS),
    #DBSCAN(eps=600, min_samples=10),
    #SpectralClustering(n_clusters=NUM_CLUSTERS, n_components=2, affinity='nearest_neighbors'),
    AgglomerativeClustering(n_clusters=NUM_CLUSTERS),
]

if __name__ == '__main__':
    df = pd.read_csv('data/data.csv').sample(20000).drop('caseid', axis=1)
    X = df.values

    plot_clusters(X, MODELS)