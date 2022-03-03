import pandas as pd
import numpy as np
from prince.famd import FAMD
import matplotlib.pyplot as plt
import seaborn as sns
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

df = pd.read_csv('data/online-shoppers-intention.csv') \
    .astype({
        'Administrative': int,
        'Administrative_Duration': float,
        'Informational': int,
        'Informational_Duration': float,
        'ProductRelated': int,
        'ProductRelated_Duration': float,
        'BounceRates': float,
        'ExitRates': float,
        'PageValues': float,
        'SpecialDay': float,
        'Month': object,
        'OperatingSystems': object,
        'Browser': object,
        'Region': object,
        'TrafficType': object,
        'VisitorType': object,
        'Weekend': object,
        'Revenue': object
    }) \
    .drop('Revenue', axis=1)

inertias = []

for n in range(2, 65):
    logging.info(f'Running FMAD for {n} components')

    famd = FAMD(n_components=n, n_iter=10)
    famd.fit(df)
    inertias.append(famd.total_inertia_)

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.lineplot(x=range(2, 65), y=inertias)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel('Number of dimensions')
plt.ylabel(f'Explained variance inertia')
plt.savefig('plots/FAMD_Inertia.svg')