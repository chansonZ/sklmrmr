def by_mrmr(X, y, n_features_to_select=None, only_get_index=True):
    from sklmrmr import MRMR
    mrmr = MRMR(n_features_to_select=n_features_to_select)
    mrmr.fit(X, y)
    index = ret.get_support(indices=True)
    if only_get_index == True:
        return index
    else:
        return mrmr.transform(X, y)
