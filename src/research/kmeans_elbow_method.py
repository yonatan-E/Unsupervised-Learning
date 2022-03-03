from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import OneHotEncoder, minmax_scale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, time

from src.utils import encode_mixed_data
from src.constants import *

assert len(sys.argv) > 1

dataset = sys.argv[1]

np.random.seed(int(time.time()))

if dataset == 'census':
    df = pd.read_csv('data/census.csv').drop(['dAge', 'dHispanic', 'iYearwrk', 'iSex', 'caseid'], axis=1)
    X = OneHotEncoder().fit_transform(df.sample(20000)).toarray()
elif dataset == 'shoppers':
    df = pd.read_csv('data/online-shoppers-intention.csv') \
        .astype(SHOPPERS_DATA_TYPES) \
        .drop(EXTERNAL_SHOPPERS_FEATURES, axis=1)
    X = encode_mixed_data(df)

losses = []
for k in range(2, 15):
    model = KMeans(n_clusters=k)
    model.fit(X)
    losses.append(model.inertia_)

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.lineplot(x=range(2, 15), y=losses)
plt.grid(axis='both', alpha=.3)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel('Number of clusters')
plt.ylabel('KMeans cost')
plt.savefig(f'plots/{dataset}/KMeans_elbow_method.svg')