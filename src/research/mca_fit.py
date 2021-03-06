import pandas as pd
from prince.mca import MCA
import matplotlib.pyplot as plt
import seaborn as sns
import logging, sys

from src.constants import EXTERNAL_CENSUS_FEATURES

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

df = pd.read_csv('data/census.csv').sample(200000).drop(EXTERNAL_CENSUS_FEATURES + ['caseid'], axis=1)

inertias = []

for n in range(2, 65):
    logging.info(f'Running MCA for {n} components')

    mca = MCA(n_components=n, random_state=0)
    mca.fit(df)
    inertias.append(mca.total_inertia_)

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.lineplot(x=range(2, 65), y=inertias)
plt.xticks(fontsize=7, alpha=.7)
plt.yticks(fontsize=7, alpha=.7)
plt.xlabel('Number of dimensions')
plt.ylabel(f'Explained variance inertia')
plt.savefig('plots/MCA_Inertia.svg')