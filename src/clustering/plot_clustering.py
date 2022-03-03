import pandas as pd
import numpy as np
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from prince.mca import MCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys, time

from src.utils import encode_mixed_data
from src.constants import *

assert len(sys.argv) > 1

dataset = sys.argv[1]

np.random.seed(int(time.time()))

if dataset == 'census':
    df = pd.read_csv('data/census.csv').drop(EXTERNAL_CENSUS_FEATURES + ['caseid'], axis=1)
    X = OneHotEncoder().fit_transform(df.sample(20000)).toarray()
elif dataset == 'shoppers':
    df = pd.read_csv('data/online-shoppers-intention.csv') \
        .astype(SHOPPERS_DATA_TYPES) \
        .drop(EXTERNAL_SHOPPERS_FEATURES, axis=1)
    X = encode_mixed_data(df)

model = KMeans(n_clusters=2)
labels = model.fit_predict(X)

#embedder = MCA(n_components=2)
embedder = TSNE(n_components=2, perplexity=30)
points = embedder.fit_transform(X)

ax = plt.gca()
ax.set_axisbelow(True)
ax.grid(True, linestyle='--')
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.scatter(points[:, 0], points[:, 1], c=labels, s=10, alpha=.4)

#plt.show()
plt.savefig(f'plots/{dataset}/{model.__class__.__name__}.png')
plt.savefig(f'plots/{dataset}/{model.__class__.__name__}.svg')