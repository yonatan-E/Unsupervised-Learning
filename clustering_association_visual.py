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
import time

from constants import *

np.random.seed(int(time.time()))

def plot_clustering_association(points, labels, feature, axe):
    for l in np.unique(labels):
        cluster = points[labels==l]
        center = np.mean(cluster, axis=0)

        for point in cluster:
            x = [point[0], center[0]]
            y = [point[1], center[1]]

            axe.plot(x, y, '#e8e2c6', alpha=0.016, linewidth=2)

    axe.set_axisbelow(True)
    axe.grid(True, linestyle='--')
    axe.set_yticklabels([])
    axe.set_xticklabels([])

    scatter = axe.scatter(points[:, 0], points[:, 1], c=feature, s=10, alpha=.4)
    legend = axe.legend(*scatter.legend_elements(), loc='upper right', title='Class')
    axe.add_artist(legend)

df = pd.read_csv('data/census-data.csv')
sample_df = df.sample(SAMPLE_SIZE)

enc = OneHotEncoder()
X = enc.fit_transform(sample_df.drop(EXTERNAL_FEATURES + ['caseid'], axis=1)).toarray()

embedder = TSNE(n_components=2, perplexity=60)
#embedder = MCA(n_components=2)
points = embedder.fit_transform(X)

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

model = KMeans(n_clusters=4)
labels = model.fit_predict(X)

plot_clustering_association(points, labels, sample_df['dAge'], axs[0])
axs[0].set_title('K-Means clusters association with dAge external feature')

plot_clustering_association(points, labels, sample_df['iYearwrk'], axs[1])
axs[1].set_title('K-Means clusters association with iYearwrk external feature')

plt.savefig('plots/clustering_dAge_iYearwrk_association.png')