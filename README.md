# ClusterUtils

The following are a few important updates to the skeleton for Homework 1.

You should either (a) Merge the latest changes from GitHub or (b) Implement the changes detailed below under "Updates."

Github repository: https://github.com/twillkens/ClusterUtils

Additionally, there are a few new "minified" datasets included in the GitHub repo. You may find these useful when first implementing your algorithms, as the amount of data is small enough to trace what is happening step-by-step.

1. three_globs_mini.csv (for KMeans)
2. squares_mini.csv (for DBSCAN)
3. eye_mini.csv (for Kernel/Spectral)


UPDATES

---------
File: sample_driver.py
Issue: KMeans cluster numbers to search for not being updated in loop.
Fix: Lines 23-26, change to:

```python
for i in range(2, 10):
    km.n_clusters = i # IMPORTANT -- Update the number of clusters.
    dfs.append(km.fit_predict_from_csv())
    cs.append(i)

# (etc.)
```

---------
File: KMeans.py
Issue: Parameter missing for choosing Lloyd's or Hartigan's algorithm.
Fix #1: Line 9, change to:

```python
def k_means(X, n_clusters=3, init='random', algorithm='lloyds', n_init=1, max_iter=300, verbose=False):
```

Fix #2: Line 60-64, change to:

```python
    def __init__(self, n_clusters=3, init='random', algorithm='lloyds', n_init=1, max_iter=300,
                 csv_path=None, keep_dataframe=True, keep_X=True, verbose=False):
        self.n_clusters = n_clusters
        self.init = init
        self.algorithm = algorithm # IMPORTANT -- attach new 'algorithm' parameter to self
        # (etc.)
```

---------
File: ClusterPlotter.py
Issue: Silhouette Index does not plot on all platforms
Fix: Method 'def \_plot\_silhouette\_', line 79, change to:

```python
def _plot_silhouette_(silhouette_table, save=False, n='silhouette_plot'):
    fig, ax = plt.subplots()
    ax.margins(0.05)
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Silhouette Index')
    plt.plot(silhouette_table['CLUSTERS'], silhouette_table['SILHOUETTE_IDX'], label='Silhouette Index')
    plt.legend()
    execute_plot(n, save)
```

---------
File: ExternalValidator.py
Issue: Centroid rows from KMeans not being dropped before calculating scores
Fix: Lines 43-45, change to:

```python
    def __init__(self, df = None, true_labels = None, pred_labels = None):
        df = df.drop('CENTROID', axis=0) # IMPORTANT -- Drop centroid rows before processing
        self.DF = df
        # (etc.)
```

---------

