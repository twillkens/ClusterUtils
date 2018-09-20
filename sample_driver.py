from ClusterUtils import DBScan
from ClusterUtils import KMeans
from ClusterUtils import InternalValidator
from ClusterUtils import ExternalValidator


db = DBScan(eps=0.3, min_points=10, csv_path='rockets.csv')
db.fit_from_csv()
db.show_plot()
db.save_plot()
db.save_csv()

km = KMeans(init='random', n_clusters=3, csv_path='three_globs.csv')
km.fit_from_csv()
km.show_plot()
km.save_plot()
km.save_csv()


km = KMeans(init='random', csv_path='well_separated.csv')
dfs = []
cs = []
for i in range(2, 10):
    km.n_clusters = i # IMPORTANT -- Update the number of clusters to run.
    dfs.append(km.fit_predict_from_csv())
    cs.append(i)

iv = InternalValidator(dfs, cluster_nums=cs)
iv.make_cvnn_table()
iv.show_cvnn_plot()
iv.save_cvnn_plot()

iv.make_silhouette_table()
iv.show_silhouette_plot()
iv.save_silhouette_plot()

iv.save_csv(cvnn=True, silhouette=True)

db = DBScan(eps=0.3, min_points=10, csv_path='rockets.csv')
data = db.fit_predict_from_csv()

ev = ExternalValidator(data)
nmi = ev.normalized_mutual_info()
nri = ev.normalized_rand_index()
a = ev.accuracy()

print([nmi, nri, a])
