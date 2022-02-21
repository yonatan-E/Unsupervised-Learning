import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

MIN_SAMPLES = 50

df = pd.read_csv('data/data.csv').sample(20000).drop('caseid', axis=1)

X = df.sample(20000).values

neigh = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X)
distances, indices = neigh.kneighbors(X)

avg_distances = distances.mean(axis=1)
avg_distances = np.sort(avg_distances)

plt.plot(avg_distances, color='blue')
plt.grid(axis='both', alpha=.3)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel('Point')
plt.ylabel(f'Avg distance to {MIN_SAMPLES} nearest neighbors')
plt.show()