import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder

MODELS = [
    GaussianMixture(n_components=3),
    #KMeans(n_clusters=21),
    #DBSCAN(eps=5000, min_samples=70),
    #SpectralClustering(n_clusters=NUM_CLUSTERS, n_components=2, affinity='nearest_neighbors'),
    #AgglomerativeClustering(n_clusters=NUM_CLUSTERS),
]

df = pd.read_csv('data/census-data.csv').sample(20000)
encoder = OneHotEncoder()

X = encoder.fit_transform(df.drop(['dAge', 'dHispanic', 'iYearwrk', 'iSex', 'caseid'], axis=1)).values

n = np.ceil(np.sqrt(len(MODELS) + 1)).astype(int)

_, axs = plt.subplots(n, n)

for idx, model in enumerate(MODELS):
    feat, labels = pd.DataFrame({'feat': df['dAge'], 'label': model.fit_predict(X)}).sort_values('label').T.values

    mutual_info = mutual_info_score(feat, labels)
    print(f'mutual info for {model}: {mutual_info}')

    y = [len(np.where(feat[:jdx]==age)[0]) * 10 for jdx, age in enumerate(feat)]

    axs[int(idx / n), idx % n].scatter(feat, y, c=labels)
    axs[int(idx / n), idx % n].set_title(model.__class__.__name__)

for ax in axs.flat:
    ax.label_outer()

plt.show()