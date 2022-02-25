from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prince.mca import MCA

EXTERNAL_FEATURES = ['dAge', 'dHispanic', 'iYearwrk', 'iSex']

df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)

mca = MCA(n_components=20, random_state=0)

X = mca.fit_transform(df.sample(20000)).values

losses = []
for k in range(2, 20):
    model = KMeans(n_clusters=k)
    model.fit(X)
    losses.append(model.inertia_)

sns.set_theme(style="darkgrid")
sns.lineplot(x=range(2, 20), y=losses)
plt.grid(axis='both', alpha=.3)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel('Number of clusters')
plt.ylabel('KMeans cost')
plt.show()