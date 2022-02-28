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

NUM_ITERATIONS = 20

MODELS = [
    GaussianMixture(n_components=3),
    KMeans(n_clusters=4),
    DBSCAN(eps=4.5, min_samples=720),
    SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=0),
    AgglomerativeClustering(n_clusters=3),
]

if __name__ == '__main__':

    results = {}

    for model in MODELS:
        results[type(model).__name__] = []
        for var in EXTERNAL_CENSUS_FEATURES:
            data = pd.read_csv(f'results/clusters_{var}_mutual_info.csv')
            mi = data[str(model)].mean()
            results[type(model).__name__].append(mi)
        logging.info(f'\n\nperforming tests for {model}...')
        
    df = pd.DataFrame(results, index=EXTERNAL_CENSUS_FEATURES)
    df['External Feature'] = df.index
    df.index = range(0, 4)
    df = df.melt(id_vars=["External Feature"], var_name="Clustering Method", value_name="Mutual Info")

    print(df)

    sns.set_theme(style="ticks")
    sns.catplot(x="External Feature", y="Mutual Info", hue="Clustering Method", kind="bar", data=df, height=8.27, aspect=11.7/8.27)

    plt.savefig("plots/cluster_association.svg")