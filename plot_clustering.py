import pandas as pd
import numpy as np
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
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
y_pred = model.fit_predict(X)

embedder = TSNE(n_components=2, perplexity=30)
X_transformed = embedder.fit_transform(X)

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.scatterplot(
    x=[x[0] for x in X_transformed], 
    y=[x[1] for x in X_transformed], 
    hue=y_pred,
    palette=sns.color_palette('tab10', 3)
)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.legend([],[], frameon=False)
plt.show()
plt.savefig(f'plots/{model.__class__.__name__}.png')