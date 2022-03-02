import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import sys, time

from utils import encode_mixed_data
from constants import *

assert len(sys.argv) > 1

dataset = sys.argv[1]

np.random.seed(int(time.time()))

if dataset == 'census':
    df = pd.read_csv('data/census.csv').drop(['dAge', 'dHispanic', 'iYearwrk', 'iSex', 'caseid'], axis=1)
    X = OneHotEncoder().fit_transform(df.sample(20000)).toarray()
    min_samples = 720
elif dataset == 'shoppers':
    df = pd.read_csv('data/online-shoppers-intention.csv') \
        .astype(SHOPPERS_DATA_TYPES) \
        .drop(EXTERNAL_SHOPPERS_FEATURES, axis=1)
    X = encode_mixed_data(df)
    min_samples = 150

neigh = NearestNeighbors(n_neighbors=min_samples).fit(X)
distances, indices = neigh.kneighbors(X)
avg_distances = np.mean(distances, axis=1)
avg_distances = np.sort(avg_distances)

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.lineplot(x=range(len(avg_distances)), y=avg_distances)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel('Point')
plt.ylabel(f'Avg distance to {min_samples} nearest neighbors')

plt.savefig(f'plots/{dataset}/k-neighbors.svg')