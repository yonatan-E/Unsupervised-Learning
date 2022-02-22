import numpy as np
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind, f_oneway
from itertools import  combinations, product
import matplotlib.pyplot as plt
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

from constants import SIGNIFICANCE_LEVEL

def find_best_model(samples_df):
    models = samples_df.columns

    best_model = models[0]

    for model in models[1:]:
        stat, p_val = ttest_ind(samples_df[model], samples_df[best_model])

        logging.info(f'T-test p-val for {model} | {best_model}: {p_val}')

        significance = 1 - p_val / 2 if stat > 0 else p_val
        if significance >= SIGNIFICANCE_LEVEL:
            best_model = model

            logging.info(f'Rejecting null hypothesis: {model} is better than {best_model}')

        logging.info(f'Accepting null hypothesis: {best_model} is better than {model}')

    return best_model

def perform_statistical_tests(samples_df, metric):
    stat, p_val = f_oneway(*samples_df.T.values)

    if 1 - p_val <= SIGNIFICANCE_LEVEL:
        logging.info(f'Anova p-val for {metric}: {p_val}')
        logging.info(f'Accepting anova null hypothesis for {metric}')

        return

    logging.info(f'Rejecting anova null hypothesis for {metric}')

    best_model = find_best_model(samples_df)

    logging.info(f'Best model for {metric}: {best_model}')

def plot_clusters(X, models, embedder=TSNE(n_components=2, perplexity=30)):
    X_transformed = embedder.fit_transform(X)

    n = np.ceil(np.sqrt(len(models) + 1)).astype(int)

    _, axs = plt.subplots(n, n)

    for idx, model in enumerate(models):
        logging.info('start')
        labels = model.fit_predict(X)
        logging.info('end')
        print(np.unique(labels))
        axs[int(idx / n), idx % n].scatter([x[0] for x in X_transformed], [x[1] for x in X_transformed], c=labels, s=10)
        axs[int(idx / n), idx % n].set_title(model.__class__.__name__)

    for ax in axs.flat:
        ax.label_outer()

    plt.show()

def delta(ck, cl):
    values = np.ones([len(ck), len(cl)])*10000
    
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])
            
    return np.min(values)
    
def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])
    
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])
            
    return np.max(values)
    
def calculate_dunn_index(X, y):
    k_list = [X[y==l] for l in np.unique(y) if l != -1]

    deltas = np.ones([len(k_list), len(k_list)])*1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])
        
        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di