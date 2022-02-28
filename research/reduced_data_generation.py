import pandas as pd
from prince.mca import MCA

df = pd.read_csv('data/census.csv').drop(['dAge', 'dHispanic', 'iYearwrk', 'iSex', 'caseid'], axis=1)
mca = MCA(n_components=60, random_state=0)
mca.fit_transform(df).to_csv('data/reduced-census.csv', index=False)