import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import logging, sys, time
import seaborn as sns
import matplotlib.pyplot as plt

from utils import perform_statistical_tests
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
        model_df = pd.DataFrame({})
        results[type(model).__name__] = []
        for var in EXTERNAL_FEATURES:
            data = pd.read_csv(f'results/clusters_{var}_mutual_info.csv')
            model_df[var] = data[str(model)]
            mi = data[str(model)].mean()
            results[type(model).__name__].append(mi)
        logging.info(f'\n\nperforming tests for {model}...')
        perform_statistical_tests(model_df, metric=str(model))
        
    df = pd.DataFrame(results, index=EXTERNAL_FEATURES)
    df['External Feature'] = df.index
    df.index = range(0, 4)
    df = df.melt(id_vars=["External Feature"], var_name="Clustering Method", value_name="Mutual Info")

    sns.set_theme(style="ticks")
    sns.catplot(x="Clustering Method", y="Mutual Info", hue="External Feature", kind="bar", data=df, height=8.27, aspect=11.7/8.27)
    

    plt.savefig("plots/variable_association.svg")