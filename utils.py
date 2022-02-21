import numpy as np
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind, f_oneway
from itertools import  combinations, product
import matplotlib.pyplot as plt

from constants import SIGNIFICANCE_LEVEL

def find_best_model(samples_df):
    models = samples_df.columns

    best_model = models[0]

    for model in models[1:]:
        stat, p_val = ttest_ind(samples_df[model], samples_df[best_model])

        print(f'T-test p-val for {model} | {best_model}: {p_val}')

        significance = 1 - p_val / 2 if stat > 0 else p_val
        if significance >= SIGNIFICANCE_LEVEL:
            best_model = model

            print(f'Rejecting null hypothesis: {model} is better than {best_model}')

        print(f'Accepting null hypothesis: {best_model} is better than {model}')

    return best_model

def perform_statistical_tests(samples_df, metric):
    stat, p_val = f_oneway(*samples_df.T.values)

    if 1 - p_val <= SIGNIFICANCE_LEVEL:
        print(f'Anova p-val for {metric}: {p_val}')
        print(f'Accepting anova null hypothesis for {metric}')

        return

    print(f'Rejecting anova null hypothesis for {metric}')

    best_model = find_best_model(samples_df)

    print(f'Best model for {metric}: {best_model}')

def calculate_dunn_index(X, y):
    try:
        clusters = [X[y==l] for l in np.unique(y) if l != -1]

        max_cluster_diameter = np.max([np.max([np.linalg.norm(a - b) for a, b in combinations(c, 2)], axis=0) for c in clusters])
        min_clusters_distance = np.min([
            np.min([
                np.linalg.norm(a - b) for a, b in product(clusters[i], clusters[j])
            ], axis=0) for i, j in combinations(range(len(clusters)), 2) if i != j
        ])

        return min_clusters_distance / max_cluster_diameter
    except:
        pass

def plot_clusters(X, models):
    X_transformed = TSNE(n_components=2, perplexity=20).fit_transform(X)

    n = np.ceil(np.sqrt(len(models) + 1)).astype(int)

    _, axs = plt.subplots(n, n)

    for idx, model in enumerate(models):

        axs[int(idx / n), idx % n].scatter([x[0] for x in X_transformed], [x[1] for x in X_transformed], c=model.fit_predict(X), s=10)
        axs[int(idx / n), idx % n].set_title(model.__class__.__name__)

    for ax in axs.flat:
        ax.label_outer()

    plt.show()