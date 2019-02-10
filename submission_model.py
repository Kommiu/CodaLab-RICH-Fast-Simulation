import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer



class Model:
    def __init__(self, n_bins=15, model_outliers=True):
        self.n_bins = n_bins
        self.model_outliers = model_outliers

    def fit(self, X, Y):
        X = X.copy()
        Y = Y.copy()

        self.x_cols = X.columns.tolist()
        self.y_cols = Y.columns.tolist()

        
        if self.model_outliers:
            mask1 = (Y == -999).values.all(axis=1)
            mask2 = (Y == 0).values.all(axis=1)
            mask = (mask1 | mask2)
        
            Y = Y[~mask]
            X = X[~mask] 

            self.probs1 = mask.mean()
            self.probs2 = mask1.mean()/mask.mean()
        
        self.binarizer = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot-dense')
        
    
        
        self.ohe_cols = ['{}_{}'.format(a, b+1) 
                for a,b 
                in zip(np.repeat(self.x_cols,self.n_bins), np.tile(range(self.n_bins), len(self.x_cols)))
                ] 
        
        labels = self.binarizer.fit_transform(X)
        labels = pd.DataFrame(labels, columns = self.ohe_cols, dtype=np.int8, index=X.index)
        
        train = pd.concat([labels, Y], axis=1)
        
        self.gmean = Y.mean()
        self.gstd = Y.std()

        self.means = train.groupby(self.ohe_cols, as_index=False).mean()
        self.stds = train.groupby(self.ohe_cols, as_index=False).std(ddof=0)
        
    
    def predict(self, X):
        
        labels = self.binarizer.transform(X)
        labels = pd.DataFrame(labels, columns=self.ohe_cols, dtype=np.int8, index=X.index)
        
        means = pd.merge(labels, self.means, how='left')[self.y_cols].fillna(self.gmean)
        stds = pd.merge(labels, self.stds, how='left')[self.y_cols].fillna(self.gstd)
        

        pred = np.random.normal(loc=means, scale=stds)
        if self.model_outliers:
            step1 = np.random.binomial(1, self.probs1,(len(X),1))
            step2 = np.random.binomial(1, self.probs2, (len(X),1))
            pred = -999*step1*step2 + (1-step1)*pred
        
        return pd.DataFrame(pred, columns = self.y_cols, index=X.index)
