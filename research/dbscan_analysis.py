import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from prince.mca import MCA
import matplotlib.pyplot as plt
import seaborn as sns

EXTERNAL_FEATURES = ['dAge', 'dHispanic', 'iYearwrk', 'iSex']

MIN_SAMPLES = 120

df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)

mca = MCA(n_components=60, random_state=0)

X = mca.fit_transform(df.sample(20000)).values

neigh = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X)
distances, indices = neigh.kneighbors(X)

avg_distances = np.mean(distances, axis=1)
avg_distances = np.sort(avg_distances)

sns.set_theme(style="darkgrid")
sns.lineplot(x=range(len(avg_distances)), y=avg_distances)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel('Point')
plt.ylabel(f'Avg distance to {MIN_SAMPLES} nearest neighbors')
plt.savefig('plots/DBSCAN_evaluation.svg')