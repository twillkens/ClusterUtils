import pandas as pd
import numpy as np
import time

# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class SuperCluster(object):

    def save_csv(self, df = None):
        if self.keep_dataframe and hasattr(self, 'DF'):
            df = self.DF
        elif self.keep_X:
            df = self.create_df()
        else:
            print('No data to save.')
            return
        dataset = self.csv_path.split(".")[0]
        timestamp = str(round(time.time()))
        filename = dataset + '_' + timestamp + '.csv'
        df.to_csv(filename)
        print('Cluster data saved as: ' + filename)
        return df

    def predict(self, X):
        if self.keep_dataframe and hasattr(self, 'DF'):
            return self.DF
        else:
            return self.labels

    def fit_predict(self, X):
        if self.keep_dataframe:
            self.fit(X)
            return self.create_df()
        else:
            return self.fit(X).labels

    def fit_from_csv(self):
        df = pd.read_csv(self.csv_path, index_col=0)
        if self.keep_dataframe:
            self.DF = df
            self.fit(df.values)
            self.create_df()
        else:
            self.fit(df.values)

        return self

    def fit_predict_from_csv(self):
        if self.keep_dataframe:
            return self.fit_from_csv().DF
        else:
            return self.fit_from_csv().labels

    def get_df(self):
        return self.DF

    def create_df(self):
        if hasattr(self, 'DF'):
            df = self.DF
            df['CLUSTER'] = self.labels
            if hasattr(self, 'centroids'):
                for c in self.centroids:
                    c = np.asarray(c)
                    d = np.append(c, -1)
                    s = pd.Series(d, index=df.columns, name='CENTROID')
                    df = df.append(s)
            self.DF = df
        elif hasattr(self, 'centroids'):
            a = np.vstack([np.asarray(self.X), self.centroids])
            l = np.asarray(self.labels)
            c_l = np.full((len(self.centroids)), -1)
            data = np.column_stack([a, np.append(l, c_l)])
            df = pd.DataFrame(data)
            self.DF = df
        else:
            data = np.column_stack([self.X, self.labels])
            df = pd.DataFrame(data)
            self.DF = df

        return df
