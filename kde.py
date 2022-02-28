import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import OneHotEncoder
from scipy.stats.mstats import mquantiles
from prince.mca import MCA
import matplotlib.pyplot as plt
import seaborn as sns
import time

np.random.seed(int(time.time()))

df = pd.read_csv('data/census.csv').drop(['dAge', 'dHispanic', 'iYearwrk', 'iSex', 'caseid'], axis=1)

X = OneHotEncoder().fit_transform(df.sample(5000)).toarray()
kde = KernelDensity(kernel='gaussian')
kde.fit(X)
X_kde = np.sort(kde.score_samples(X))

tau = mquantiles(X_kde, 0.98)

labels = [1 if density <= tau else 2 for density in X_kde]
mca = MCA(n_components=2)
points = mca.fit_transform(X).values

ax = plt.gca()
ax.set_axisbelow(True)
ax.grid(True, linestyle='--')
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.scatter(points[:, 0], points[:, 1], c=labels, s=10, alpha=.4)

plt.show()

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.lineplot(x=range(len(X_kde)), y=X_kde)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.axhline(y=tau, linestyle='--')
plt.xlabel('Point')
plt.ylabel('Density')
plt.show()