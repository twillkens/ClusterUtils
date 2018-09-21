# ClusterUtils

## Update
The following are a few important updates to the skeleton for Homework 1.

You should either:
* Merge the latest changes from GitHub (preferred), or
* Implement the changes detailed below under "Changes" into your skeleton.

## New Datasets

There are a few new "minified" datasets included in the GitHub repo. You may find these useful when first implementing your algorithms, as the amount of data is small enough to trace what is happening step-by-step.

They can be found at: https://github.com/twillkens/ClusterUtils/tree/master/Mini_Datasets
1. three\_globs\_mini.csv (for KMeans)
2. squares\_mini.csv (for DBSCAN)
3. eye\_mini.csv (for Kernel/Spectral)

Additionally, there are extra datasets to try out once your code is more mature. These include more challenging cases.

They can be found at: https://github.com/twillkens/ClusterUtils/tree/master/Extra_Datasets
1. iris.csv (classic dataset with three classes in four dimensions, 150 samples)
2. globs\_large.csv (larger version of three\_globs, 1040 samples)
3. globs\_huge.csv (2081 samples, overlapping data)
4. fifteen\_clusters (fifteen clusters, 5000 samples)
5. 32\_dim\_16\_clusters.csv (32 dimensions, 16 clusters, and 1024 samples)
6. spiral.csv (312 samples, not linearally separable)

## Misc. Notes
1. It is recommended to implement a simple random KMeans before implementing the Validators. This way, you have useful, real data to work with.
2. Implementation of KernelKM.py and Spectral.py now are both bonus questions, worth 5 points each. It is recommended to attempt them, however, as they are very interesting.


## Changes

---------
File: sample\_driver.py  
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

