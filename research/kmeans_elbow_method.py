from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import numpy as np

EXTERNAL_FEATURES = ['dAge', 'dHispanic', 'iYearwrk', 'iSex']

df = pd.read_csv('data/census-data.csv').drop(EXTERNAL_FEATURES + ['caseid'], axis=1)

X = df.sample(20000).values

model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 25))

visualizer.fit(X)
visualizer.show()