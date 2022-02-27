import pandas as pd
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
fig.suptitle('A single ax with no data')

fig.show()