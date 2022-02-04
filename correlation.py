import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/data.csv', nrows=10000)

feat1 = df['dHour89']
feat2 = df['dIncome1']

print(f'Correlation: {feat1.corr(feat2)}')

regressor = LinearRegression().fit(feat1.values.reshape(-1, 1), feat2.values.reshape(-1, 1))

plt.scatter(feat1, feat2, color='red', s=3)
plt.plot(feat1, regressor.predict(feat1.values.reshape(-1, 1)), color='blue')
plt.xlabel(feat1.name)
plt.ylabel(feat2.name)
plt.show()