import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 



def make_mask(x, bounds):
    lefts = bounds[:-1]
    lefts[0] -= 0.1
    rights = bounds[1:]
    
    a = np.repeat(x, len(rights)).reshape((-1,len(rights))) <= rights
    b = np.repeat(x, len(lefts)).reshape((-1,len(lefts))) > lefts
    
    return np.where(a & b)[1]

def f(x, x_cols):
    d = {}
    for col in x_cols:
        d[col] = x[col].median()
    for col in y_cols:
        d[col + '_mean'] = x[col].mean()
        d[col + '_std'] =  x[col].std()

    return pd.Series(d,list(d.keys()))

class Model:
    
    def __init__(self, n_intervals = 5):
        self.n_intervals = n_intervals 

    def fit(self, X, Y):
        self.n_intervals = 5
        x_cols = X.columns.tolist()
        y_cols = Y.columns.tolist()
        
        data = pd.concat([X.copy(), Y.copy()], axis=1)
        masks = pd.DataFrame(index=X.index, columns=X.columns)
        
        quants = [i/self.n_intervals for i in range(self.n_intervals + 1)]
        bounds =  np.quantile(X, quants, axis=0)
        
        for i,col in enumerate(X):
            masks[col] = make_mask(X[col].values, bounds[:,i]).astype(str)
        
        
        data['mask'] = masks.sum(axis=1)
        
        train = data.groupby('mask').apply(lambda x: f(x, x_cols))

        
        self.means = LinearRegression()
        self.stds = LinearRegression()
        
    
        self.means.fit(train[x_cols].values, train[[ col + '_mean' for col in y_cols]].values)
        self.stds.fit(train[x_cols].values, train[[col + '_std'  for col in y_cols]].values)
        
        self.y_cols = y_cols
    def predict(self, X):
        prediction = pd.DataFrame()
        means = self.means.predict(X.values)
        stds = self.stds.predict(X.values)
        pred = np.random.standard_normal()*stds + means
        return pd.DataFrame(pred, columns = self.y_cols)


