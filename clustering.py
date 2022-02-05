import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind, f_oneway
import numpy as np
from itertools import combinations, product
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

NUM_CLUSTERS = 9
SIGNIFICANCE_LEVEL = 0.99
NUM_ITERATIONS = 20
BATCH_SIZE = 200

MODELS = [
    GaussianMixture(n_components=NUM_CLUSTERS),
    KMeans(n_clusters=NUM_CLUSTERS),
    DBSCAN(eps=100, min_samples=8),
    SpectralClustering(n_clusters=NUM_CLUSTERS, n_components=2, affinity='nearest_neighbors'),
    AgglomerativeClustering(n_clusters=NUM_CLUSTERS),
]

def plots(X):
    X_transformed = TSNE(n_components=2, perplexity=20).fit_transform(X)

    n = np.ceil(np.sqrt(len(MODELS) + 1)).astype(int)

    _, axs = plt.subplots(n, n)

    for idx, model in enumerate(MODELS):

        axs[int(idx / n), idx % n].scatter([x[0] for x in X_transformed], [x[1] for x in X_transformed], c=model.fit_predict(X), s=10)
        axs[int(idx / n), idx % n].set_title(model.__class__.__name__)

    for ax in axs.flat:
        ax.label_outer()

    plt.show()

def calculate_dunn_index(clusters):
    max_cluster_diameter = np.max([np.max([np.linalg.norm(a - b) for a, b in combinations(c, 2)], axis=0) for c in clusters])
    min_clusters_distance = np.min([
        np.min([
            np.linalg.norm(a - b) for a, b in product(clusters[i], clusters[j])
        ], axis=0) for i, j in combinations(range(len(clusters)), 2) if i != j
    ])

    return min_clusters_distance / max_cluster_diameter

def cluster_and_evaluate(X):
    evaluation_results = {'silhouette': {}, 'dunn_index': {}}

    for model in MODELS:
        y_pred = model.fit_predict(X_batch)

        evaluation_results['silhouette'][model] = silhouette_score(X_batch, y_pred)
        evaluation_results['dunn_index'][model] = calculate_dunn_index([X_batch[y_pred==l] for l in np.unique(y_pred) if l != -1])

    return evaluation_results

def find_best_model(samples_df):
    best_model = MODELS[0]

    for model in MODELS[1:]:
        stat, p_val = ttest_ind(samples_df[model], samples_df[best_model])
        significance = 1 - p_val / 2 if stat > 0 else p_val

        if significance >= SIGNIFICANCE_LEVEL:
            best_model = model

    return best_model

if __name__ == '__main__':
    df = pd.read_csv('data/data.csv', nrows=1000).drop('caseid', axis=1)

    clustering_eval_samples = {
        'silhouette': pd.DataFrame(),
        'dunn_index': pd.DataFrame()
    }

    for _ in range(NUM_ITERATIONS):
        print(f'Running iteration {_}')

        X_batch = df.sample(BATCH_SIZE).to_numpy()
        clustering_evaluation = cluster_and_evaluate(X_batch)

        clustering_eval_samples = {
            method: clustering_eval_samples[method].append(clustering_evaluation[method], ignore_index=True) 
            for method in clustering_eval_samples
        }

    for method in clustering_eval_samples:
        samples_df = clustering_eval_samples[method]

        stat, p_val = f_oneway(*samples_df.T.to_numpy())

        if p_val > SIGNIFICANCE_LEVEL:
            print(f'Accepting anova null hypothesis for {method} method, p-val: {p_val}')

            continue

        print(f'Rejecting anova null hypothesis for {method} method, p_val: {p_val}')

        best_model = find_best_model(samples_df)

        print(f'Best model for {method} test: {best_model}')

    plots(df.to_numpy())