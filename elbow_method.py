from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/data.csv', nrows=1000).drop('caseid', axis=1)
X = df.to_numpy()

model = KMeans()
visualizer = KElbowVisualizer(model, k=(4, 30))

visualizer.fit(X)
visualizer.show()