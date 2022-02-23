from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EXTERNAL_FEATURES = ['dAge', 'dHispanic', 'iYearwrk', 'iSex']

df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)

X = df.sample(20000).values

losses = []
for k in range(2, 16):
    model = KMeans(n_clusters=k)
    model.fit(X)
    losses.append(model.inertia_)

plt.plot(losses, color='blue', marker='o')
plt.grid(axis='both', alpha=.3)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel('Number of clusters')
plt.ylabel('KMeans cost')
plt.show()