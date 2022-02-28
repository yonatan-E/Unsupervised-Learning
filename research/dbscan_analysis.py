import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time

np.random.seed(int(time.time()))

df = pd.read_csv('data/census-data.csv').drop(['dAge', 'dHispanic', 'iYearwrk', 'iSex', 'caseid'], axis=1)

enc = OneHotEncoder()

X = enc.fit_transform(df.sample(20000))

sns.set_style("darkgrid", {"axes.facecolor": ".9"})

neigh = NearestNeighbors(n_neighbors=720).fit(X)
distances, indices = neigh.kneighbors(X)
avg_distances = np.median(distances, axis=1)
avg_distances = np.sort(avg_distances)

sns.lineplot(x=range(len(avg_distances)), y=avg_distances)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel('Point')
plt.ylabel('Avg distance to 720 nearest neighbors')
plt.show()

plt.savefig('plots/k-neighbors.svg')