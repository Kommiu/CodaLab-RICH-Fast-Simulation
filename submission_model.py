import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer




class Model:
    def __init__(self, n_bins=7):
        self.n_bins = n_bins

    def fit(self, X, Y):
        
        self.x_cols = X.columns.tolist()
        self.y_cols = Y.columns.tolist()
        
        
        self.binarizer = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot-dense')
        
    
        
        self.ohe_cols = ['{}_{}'.format(a, b+1) 
                for a,b 
                in zip(np.repeat(self.x_cols,self.n_bins), np.tile(range(self.n_bins), len(self.x_cols)))
                ] 
        
        train = self.binarizer.fit_transform(X)
        train = pd.DataFrame(train, columns = self.ohe_cols, dtype=np.int8)
        
        train = pd.concat([train, Y], axis=1)
        self.global_means = Y.mean()
        
        self.means = train.groupby(self.ohe_cols).mean().reset_index()
        self.stds = train.groupby(self.ohe_cols).std().reset_index()
        
    
    def predict(self, X):
        
        test = self.binarizer.transform(X)
        test = pd.DataFrame(test, columns=self.ohe_cols, dtype=np.int8)
        means = pd.merge(test, self.means, how='left')[self.y_cols]
        stds = pd.merge(test, self.stds, how='left')[self.y_cols]
        
        pred = means.values + stds.values*np.random.standard_normal(stds.shape)
        
        return pd.DataFrame(pred, columns = self.y_cols).fillna(self.global_means)

