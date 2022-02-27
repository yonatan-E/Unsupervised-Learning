from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

EXTERNAL_FEATURES = ['dAge', 'dHispanic', 'iYearwrk', 'iSex']

np.random.seed(int(time.time()))

df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)

enc = OneHotEncoder()

X = enc.fit_transform(df.sample(20000)).toarray()

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
plt.savefig('plots/KMeans_elbow_method.svg')