from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import MDS, LocallyLinearEmbedding, SpectralEmbedding, Isomap, TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

MODELS = (
    (TSNE(n_components=2, perplexity=30), 'TSNE'),
    (Isomap(n_components=2, n_neighbors=4), 'ISOMAP'),
    (PCA(n_components=2), 'PCA'),
    (FastICA(n_components=2), 'ICA'),
    (SpectralEmbedding(n_components=2, n_neighbors=8), 'SpectralEmbedding'),
    (MDS(n_components=2), 'MDS'),
    (LocallyLinearEmbedding(n_components=2, n_neighbors=13), 'LLE')
)

def plots(X):
    n = np.ceil(np.sqrt(len(MODELS) + 1)).astype(int)

    _, axs = plt.subplots(n, n)

    for idx, model in enumerate(MODELS):
        model, name = model

        X_transformed = model.fit_transform(X)

        axs[int(idx / n), idx % n].scatter([x[0] for x in X_transformed], [x[1] for x in X_transformed], s=10)
        axs[int(idx / n), idx % n].set_title(name)

    for ax in axs.flat:
        ax.label_outer()

    plt.show()

df = pd.read_csv('data/data.csv', nrows=1000).drop('caseid', axis=1)
X = df.to_numpy()

plots(X)