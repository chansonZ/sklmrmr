使用方法

# example

from sklearn.datasets import load_digits

from sklmrmr import MRMR

digits = load_digits()

X = digits.images.reshape((len(digits.images), -1)).astype(int)

y = digits.target

by_mrmr(X,y,n_features_to_select=5)
 
In [6]: by_mrmr(X,y,n_features_to_select=5)

Out[6]: array([21, 26, 33, 43, 61])

# wrapper 

def by_mrmr(X, y, n_features_to_select=None, only_get_index=True):
    from sklmrmr import MRMR
    mrmr = MRMR(n_features_to_select=n_features_to_select)
    mrmr.fit(X, y)
    index = ret.get_support(indices=True)
    if only_get_index == True:
        return index
    else:
        return mrmr.transform(X, y)
