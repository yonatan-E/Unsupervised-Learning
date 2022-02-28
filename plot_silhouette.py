import pandas as pd
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_silhouette(model, axe):
    silhouette_results_df = pd.read_csv(f'results/{model}_silhouette.csv')
    silhouette_scores = silhouette_results_df.mean().values

    p = sns.lineplot(x=silhouette_results_df.columns[:15], y=silhouette_scores[:15], ax=axe)
    p.set_xlabel('Number of clusters' if model != 'DBSCAN' else 'Epsilon')
    p.set_ylabel('Silhouette score')
    p.set_title(f'Silhouette score for {model}')

sns.set_style("darkgrid", {"axes.facecolor": ".9"})

f, axs = plt.subplots(2, 2, figsize=(15, 10))
f.subplots_adjust(hspace=.3)

plot_model_silhouette('KMeans', axs[0, 0])
plot_model_silhouette('GaussianMixture', axs[0, 1])
plot_model_silhouette('AgglomerativeClustering', axs[1, 0])
plot_model_silhouette('SpectralClustering', axs[1, 1])

f.savefig('plots/silhouette.svg')