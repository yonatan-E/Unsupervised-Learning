from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import numpy as np

df = pd.read_csv('data/data.csv').drop('caseid', axis=1)

X = df.sample(20000).values

model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 25))

visualizer.fit(X)
visualizer.show()