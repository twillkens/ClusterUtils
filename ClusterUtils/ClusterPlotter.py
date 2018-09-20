import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

def merge_centroids(X, centroids, labels):
        a = np.vstack([np.asarray(X), centroids])
        l = np.asarray(labels)
        c_l = np.full((len(centroids)), -1)
        return np.column_stack([a, np.append(l, c_l)])

def execute_plot(n, save):
    if save:
        filename = n + '_' + str(round(time.time())) + '.png'
        plt.savefig(filename)
        print('Plot saved as ' + filename)
    else:
        plt.show()

def reduce_dimensions(df):
    clusters = df[df.shape[1] - 1].reset_index(drop=True)
    df.drop(df.shape[1] - 1, axis='columns', inplace=True)
    reduced_data = PCA(n_components=2).fit_transform(df)
    reduced_df = pd.DataFrame(reduced_data)
    reduced_df[2] = clusters
    return reduced_df

def _plot_generic_(X=None, labels=None, df=None, save=False, n='plot'):
    if df is None:
        a = np.asarray(X)
        l = np.asarray(labels)
        df = pd.DataFrame(np.column_stack([a, l]))
    else:
        df = df.copy()

    df.columns = range(df.shape[1])

    if len(df.columns) > 3:
        df = reduce_dimensions(df)

    groups = df.groupby(df.shape[1] - 1)
    fig, ax = plt.subplots()
    ax.margins(0.05)

    for name, group in groups:
        ax.plot(group[0], group[1], marker='o', linestyle='', ms=2,
                                                label=name, zorder=0)
    execute_plot(n, save)

def _plot_kmeans_(X = None, centroids=None, labels=None, df=None,
                                    save=False, n='kmeans_plot'):
    if df is None:
        data = merge_centroids(X, centroids, labels)
        df = pd.DataFrame(data)
    else:
        df = df.copy()

    df.columns = range(df.shape[1])

    if len(df.columns) > 3:
        df = reduce_dimensions(df)

    groups = df.groupby(df.shape[1] - 1)
    fig, ax = plt.subplots()
    ax.margins(0.05)
    for _, group in groups:
        if group.iloc[0][2] == -1:
            ax.plot(group[0], group[1], marker='+', linestyle='', ms=12,
                                                    zorder=99, color='black')
        else:
            ax.plot(group[0], group[1], marker='o', linestyle='', ms=2, zorder=0)
    execute_plot(n, save)

# IMPORTANT -- Update to work on different platforms
def _plot_silhouette_(silhouette_table, save=False, n='silhouette_plot'):
    fig, ax = plt.subplots()
    ax.margins(0.05)
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Silhouette Index')
    plt.plot(silhouette_table['CLUSTERS'], silhouette_table['SILHOUETTE_IDX'], label='Silhouette Index')
    plt.legend()
    execute_plot(n, save)

def _plot_cvnn_(cvnn_table, save=False, n='cvnn_table'):
    cnn_table = cvnn_table.copy()
    low_points = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('CLUSTERS')
    ax.set_ylabel('K')
    ax.set_zlabel('CVNN')
    groups = cvnn_table.groupby('K',as_index=False)
    for name, g in groups:
        low = g.loc[g.idxmin(axis=0)['CVNN']]
        cvnn_table = cvnn_table.drop(low.name)
        ax.scatter(low['CLUSTERS'].astype('int'), low['K'].astype('int'),
                                                    low['CVNN'], c='red', s=40)
        ax.text(low['CLUSTERS'],low['K'],low['CVNN'],
                '%s' % (' ' + str(low['CLUSTERS'])), size=10,
                zorder=1,color='k')
    ax.scatter(cvnn_table['CLUSTERS'].astype('int'), cvnn_table['K'].astype(int),
                                                    cvnn_table['CVNN'], c='blue')
    execute_plot(n, save)
