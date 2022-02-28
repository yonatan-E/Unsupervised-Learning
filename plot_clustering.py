import pandas as pd
import numpy as np
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from prince.mca import MCA
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time

from constants import *

np.random.seed(int(time.time()))

df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)

enc = OneHotEncoder()
X = enc.fit_transform(df.sample(SAMPLE_SIZE)).toarray()

model = GaussianMixture(n_components=3)
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

plt.savefig('plots/GaussianMixture.png')