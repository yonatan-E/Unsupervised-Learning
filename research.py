import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering

#df = pd.read_csv('data/original-data.csv')
#from constants import EXTERNAL_FEATURES
#df.drop(EXTERNAL_FEATURES, axis=1).to_csv('data/data.csv')

df = pd.read_csv('data/data.csv', nrows=1000).drop('caseid', axis=1)

#X = TSNE(n_components=2, perplexity=20).fit_transform(df.to_numpy())
X = df.sample(200).to_numpy()

model = KMeans(n_clusters=10)
preds = model.fit_predict(X)

feat1_idx = np.where(df.columns=='dHour89')[0][0]
feat2_idx = np.where(df.columns=='dIncome1')[0][0]

plt.scatter([x[0] for x in X], [x[2] for x in X], c=preds)
plt.show()