import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import logging, sys, time
import seaborn as sns
import matplotlib.pyplot as plt

from constants import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")


np.random.seed(int(time.time()))

assert len(sys.argv) > 1

dataset = sys.argv[1]
save = len(sys.argv) > 2 and sys.argv[2] == '--save'

NUM_ITERATIONS = 20

if dataset == 'census':
    SAMPLE_SIZE = 20000
    EXTERNAL_FEATURES = EXTERNAL_CENSUS_FEATURES
    MODELS = [
        GaussianMixture(n_components=3),
        KMeans(n_clusters=4),
        DBSCAN(eps=4.5, min_samples=720),
        SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0),
        AgglomerativeClustering(n_clusters=3),
    ]
elif dataset == 'shoppers':
    SAMPLE_SIZE = 1000
    EXTERNAL_FEATURES = EXTERNAL_SHOPPERS_FEATURES
    MODELS = [
        GaussianMixture(n_components=3),
        KMeans(n_clusters=2),
        SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0),
        AgglomerativeClustering(n_clusters=2),
    ]


if __name__ == '__main__':

    results = {}

    for model in MODELS:
        results[type(model).__name__] = []
        for var in EXTERNAL_FEATURES:
            data = pd.read_csv(f'results/{dataset}/clusters_{var}_mutual_info.csv')
            mi = data[str(model)].mean()
            results[type(model).__name__].append(mi)
        logging.info(f'\n\nperforming tests for {model}...')
        
    df = pd.DataFrame(results, index=EXTERNAL_FEATURES)
    df['External Feature'] = df.index
    df.index = range(0, len(EXTERNAL_FEATURES))
    df = df.melt(id_vars=["External Feature"], var_name="Clustering Method", value_name="Mutual Info")

    sns.set_theme(style="ticks")
    sns.catplot(x="External Feature", y="Mutual Info", hue="Clustering Method", kind="bar", data=df, height=8.27, aspect=11.7/8.27)

    plt.savefig(f"plots/{dataset}/cluster_association.svg")
    plt.savefig(f"plots/{dataset}/cluster_association.png")