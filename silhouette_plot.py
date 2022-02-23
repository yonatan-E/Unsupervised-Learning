import pandas as pd
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

def plot_model_silhouette(model, axe):
    silhouette_results_df = pd.read_csv(f'results/{model}_silhouette.csv')
    silhouette_scores = silhouette_results_df.mean().values

    plt.sca(axe)
    plt.plot(silhouette_results_df.columns, silhouette_scores, color='blue', marker='o')
    plt.grid(axis='both', alpha=.3)
    plt.xticks(fontsize=7, alpha=.7)
    plt.yticks(fontsize=7, alpha=.7)
    plt.xlabel('Number of clusters' if model != 'DBSCAN' else 'Epsilon')
    plt.ylabel('Silhouette score')
    plt.title(f'Silhouette score for {model}')

f, axs = plt.subplots(3, 2, figsize=(10, 15))
f.subplots_adjust(hspace=.3)

plot_model_silhouette('KMeans', axs[0, 0])
plot_model_silhouette('GaussianMixture', axs[0, 1])
plot_model_silhouette('DBSCAN', axs[1, 0])
plot_model_silhouette('AgglomerativeClustering', axs[1, 1])
plot_model_silhouette('SpectralClustering', axs[2, 0])

f.delaxes(axs[2, 1])

plt.savefig('plots/silhouette.svg')