import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale, OneHotEncoder
from scipy.stats import ttest_ind, f_oneway
from itertools import  combinations, product
import matplotlib.pyplot as plt
import logging, sys

from constants import *

def find_best_model(samples_df):
    models = samples_df.columns

    best_model = models[0]

    for model in models[1:]:
        stat, p_val = ttest_ind(samples_df[model], samples_df[best_model])

        logging.info(f'T-test for {model} | {best_model}: p-val - {p_val}, stat - {stat}')

        significance = 1 - p_val / 2 if stat > 0 else p_val
        if significance >= SIGNIFICANCE_LEVEL:
            logging.info(f'Rejecting null hypothesis: {model} is better than {best_model}')

            best_model = model
        else:
            logging.info(f'Accepting null hypothesis: {best_model} is better than {model}')

    return best_model

def perform_statistical_tests(samples_df, metric):
    stat, p_val = f_oneway(*samples_df.T.values)

    logging.info(f'Anova p-val for {metric}: {p_val}')

    if 1 - p_val <= SIGNIFICANCE_LEVEL:
        logging.info(f'Accepting anova null hypothesis for {metric}')

        return

    logging.info(f'Rejecting anova null hypothesis for {metric}')

    best_model = find_best_model(samples_df)

    logging.info(f'Best model for {metric}: {best_model}')

def plot_clusters(X, models, embedder=TSNE(n_components=2, perplexity=30)):
    X_transformed = embedder.fit_transform(X)

    if isinstance(X_transformed, pd.DataFrame):
        X_transformed = X_transformed.values

    n = np.ceil(np.sqrt(len(models) + 1)).astype(int)
    _, axs = plt.subplots(n, n)

    for idx, model in enumerate(models):
        labels = model.fit_predict(X)

        print(f'Labels for {model}: {np.unique(labels)}')

        axs[int(idx / n), idx % n].scatter([x[0] for x in X_transformed], [x[1] for x in X_transformed], c=labels, s=10)
        axs[int(idx / n), idx % n].set_title(model.__class__.__name__)

    for ax in axs.flat:
        ax.label_outer()

    plt.show()

def encode_mixed_data(df, normalize=True):
    numeric_cols = df.select_dtypes(include=np.number).values
    categorial_cols = df.select_dtypes(include='object').values

    numeric_part = minmax_scale(numeric_cols) if normalize else numeric_cols

    categorial_one_hot = OneHotEncoder().fit_transform(categorial_cols).toarray()
    categorial_part = minmax_scale(categorial_one_hot) if normalize else categorial_one_hot

    return np.concatenate([numeric_part, categorial_part], axis=1)