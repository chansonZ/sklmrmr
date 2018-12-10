__version__ = '0.2.1' # from 0.2.0  by chanson  

__all__ = ['MRMR']

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
# new version of sklearn by chanson
from sklearn.utils import check_X_y
#from sklearn.utils import check_arrays, safe_mask
from ._mrmr import _mrmr, MAXREL, MID, MIQ


class MRMR(BaseEstimator, SelectorMixin):
    def __init__(self,
                 n_features_to_select=None,
                 method='MID',
                 normalize=False,
                 verbose=0):

        method = method.upper()
        if method not in ('MAXREL', 'MID', 'MIQ'):
            raise ValueError("method must be one of 'MAXREL', 'MID', or 'MIQ'")

        self.k = n_features_to_select
        self.method = method
        self.normalize = normalize
        self.verbose = verbose

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse='csc')

        n_samples, n_features = X.shape

        if self.k is None:
            k = n_features // 2
        else:
            k = self.k

        if self.method == 'MAXREL':
            method = MAXREL
        elif self.method == 'MID':
            method = MID
        elif self.method == 'MIQ':
            method = MIQ
        else:
            raise RuntimeError("unknown method: '{0}'".format(method))

        X_classes = np.array(list(set(X.reshape((n_samples * n_features, )))))
        y_classes = np.array(list(set(y.reshape((n_samples, )))))

        idxs, _ = _mrmr(n_samples, n_features, y.astype(np.long),
                        X.astype(np.long), y_classes.astype(np.long),
                        X_classes.astype(np.long), y_classes.shape[0],
                        X_classes.shape[0], k, method, self.normalize)

        support_ = np.zeros(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int) + k

        support_[idxs] = True
        for i, idx in enumerate(idxs, start=1):
            ranking_[idx] = i

        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self

    def _get_support_mask(self):
        return self.support_
