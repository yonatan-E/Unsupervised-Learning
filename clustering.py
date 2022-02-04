import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_rel, f_oneway
import numpy as np
from itertools import product, combinations
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

NUM_CLUSTERS = 9
SIGNIFICANCE_LEVEL = 0.01
NUM_ITERATIONS = 50
BATCH_SIZE = 200

MODELS = (
    (KMeans(n_clusters=NUM_CLUSTERS), 'KMeans'),
    (GaussianMixture(n_components=NUM_CLUSTERS), 'GaussianMixture'),
    (DBSCAN(eps=100, min_samples=8), 'DBSCAN'),
    (SpectralClustering(n_clusters=NUM_CLUSTERS, n_components=2, affinity='nearest_neighbors'), 'SpecturalClustering'),
    (AgglomerativeClustering(n_clusters=NUM_CLUSTERS), 'HierarchialClustering')
)

def plots(X):
    X_transformed = TSNE(n_components=2, perplexity=20).fit_transform(X)

    n = np.ceil(np.sqrt(len(MODELS) + 1)).astype(int)

    _, axs = plt.subplots(n, n)

    for idx, model in enumerate(MODELS):
        model, name = model

        axs[int(idx / n), idx % n].scatter([x[0] for x in X_transformed], [x[1] for x in X_transformed], c=model.fit_predict(X), s=10)
        axs[int(idx / n), idx % n].set_title(name)

    for ax in axs.flat:
        ax.label_outer()

    plt.show()

def calculate_dunn_index(clusters):
    max_cluster_diameter = np.max([np.max([np.linalg.norm(a - b) for a, b in product(c, c)], axis=0) for c in clusters])
    min_clusters_distance = np.min([
        np.min([
            np.linalg.norm(a - b) for a, b in product(clusters[i], clusters[j])
        ], axis=0) for i, j in product(range(len(clusters)), range(len(clusters))) if i != j
    ])

    return min_clusters_distance / max_cluster_diameter

if __name__ == '__main__':
    df = pd.read_csv('data/data.csv', nrows=1000).drop('caseid', axis=1)

    plots(df.to_numpy())

    tests_data = {
        'silhouette': {name: [] for _, name in MODELS},
        'dunn_index': {name: [] for _, name in MODELS}
    }

    for _ in range(NUM_ITERATIONS):
        print(f'Running iteration {_}')

        X_batch = df.sample(BATCH_SIZE).to_numpy()

        for model, name in MODELS:
            y_pred = model.fit_predict(X_batch)

            tests_data['silhouette'][name].append(silhouette_score(X_batch, y_pred))
            tests_data['dunn_index'][name].append(calculate_dunn_index(
                [[X_batch[i] for i in range(BATCH_SIZE) if y_pred[i] == l] for l in np.unique(y_pred)]
            ))

    for test in tests_data:
        stat, p_val = f_oneway(*tests_data[test].values())

        print(f'Anova p-value for {test} test: {p_val}')

        if SIGNIFICANCE_LEVEL < p_val:
            print(f'Accepting anova null hypothesis for {test} test')

            continue

        print(f'Rejecting anova null hypothesis for {test} test')

        best_model = None

        for pair in combinations(MODELS, 2):
            model_1, model_2 = pair[0][1], pair[1][1]

            stat, p_val = ttest_rel(tests_data[test][model_1], tests_data[test][model_2], alternative='greater')

            better_model = model_1 if SIGNIFICANCE_LEVEL >= p_val else  model_2

            if best_model is None:
                best_model = better_model
            else:
                stat, p_val = ttest_rel(tests_data[test][better_model], tests_data[test][best_model], alternative='greater')

                if SIGNIFICANCE_LEVEL >= p_val:
                    best_model = better_model

        print(f'Best model for {test} test: {best_model}')