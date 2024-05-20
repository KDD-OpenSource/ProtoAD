from sklearn import preprocessing


def scaling(X):
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)

    return X_minmax
