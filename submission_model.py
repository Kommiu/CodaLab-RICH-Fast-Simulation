import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer




class Model:
    def __init__(self, n_bins=7):
        self.n_bins = n_bins

    def fit(self, X, Y):
        
        self.x_cols = X.columns.tolist()
        self.y_cols = Y.columns.tolist()

        mask1 = (Y == -999).values.all(axis=1)
        mask2 = (Y == 0).values.all(axis=1)
 
        mask = (mask1 | mask2)
        
        Y = Y[~mask]
        X = X[~mask] 

        self.probs1 = mask.mean()
        self.probs2 = mask1.mean()/mask.mean()
        
        self.binarizer = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot-dense')
        
    
        
        self.ohe_cols = ['{}_{}'.format(a, b+1) 
                for a,b )
                in zip(np.repeat(self.x_cols,self.n_bins), np.tile(range(self.n_bins), len(self.x_cols)))
                ] 
        
        train = self.binarizer.fit_transform(X)
        train = pd.DataFrame(train, columns = self.ohe_cols, dtype=np.int8)
        
        train = pd.concat([train, Y], axis=1)
        
        self.means = train.groupby(self.ohe_cols).mean().reset_index().fillna(Y.mean(axis=0))
        self.stds = train.groupby(self.ohe_cols).std().reset_index().fillna(Y.std(axis=0))
        
    
    def predict(self, X):
        
        test = self.binarizer.transform(X)
        test = pd.DataFrame(test, columns=self.ohe_cols, dtype=np.int8)
        means = pd.merge(test, self.means, how='left')[self.y_cols]
        stds = pd.merge(test, self.stds, how='left')[self.y_cols]
        

        step1 = np.random.binomial(1, self.probs1,(len(X),1))
        step2 = np.random.binomial(1, self.probs2, (len(X),1))
        normal = np.random.normal(loc=means, scale=stds)
        
        pred = -999*step1*step2 + (1-step1)*normal
        
        return pd.DataFrame(pred, columns = self.y_cols)
