import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from clustering import NUM_CLUSTERS

from constants import EXTERNAL_FEATURES

NUM_CLUSTERS = 8

MODELS = [
    GaussianMixture(n_components=NUM_CLUSTERS),
    KMeans(n_clusters=NUM_CLUSTERS),
    DBSCAN(eps=100, min_samples=8),
    #SpectralClustering(n_clusters=NUM_CLUSTERS, n_components=2, affinity='nearest_neighbors'),
    AgglomerativeClustering(n_clusters=NUM_CLUSTERS),
]

df = pd.read_csv('data/original-data.csv', nrows=10000)
X = df.drop(EXTERNAL_FEATURES, axis=1)

n = np.ceil(np.sqrt(len(MODELS) + 1)).astype(int)

_, axs = plt.subplots(n, n)

for idx, model in enumerate(MODELS):
    feat, labels = pd.DataFrame({'feat': df['dAge'], 'label': model.fit_predict(X.values)}).sort_values('label').T.values

    y = [10 * len(np.where(feat[:jdx]==age)[0]) for jdx, age in enumerate(feat)]

    axs[int(idx / n), idx % n].scatter(feat, y, c=labels)
    axs[int(idx / n), idx % n].set_title(model.__class__.__name__)

for ax in axs.flat:
    ax.label_outer()

plt.show()

'''
for i in range(0, 8):
    rates = [len(np.where(feat[labels==l]==i)[0]) for l in np.unique(labels)]
    print(f'age {i}: {np.std(rates)}')
'''