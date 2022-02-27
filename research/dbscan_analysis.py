import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import random, time

EXTERNAL_FEATURES = ['dAge', 'dHispanic', 'iYearwrk', 'iSex']

MIN_SAMPLES = 720

np.random.seed(int(time.time()))

df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)

enc = OneHotEncoder()

X = enc.fit_transform(df.sample(20000))

neigh = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X)
distances, indices = neigh.kneighbors(X)

avg_distances = np.mean(distances, axis=1)
avg_distances = np.sort(avg_distances)

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.lineplot(x=range(len(avg_distances)), y=avg_distances)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel('Point')
plt.ylabel(f'Avg distance to {MIN_SAMPLES} nearest neighbors')
plt.show()