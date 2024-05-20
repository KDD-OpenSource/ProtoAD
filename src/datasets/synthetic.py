import pandas as pd
import numpy as np
from .dataset import Dataset


class Synthetic(Dataset):
    """0 is the outlier class. The training set is free of outliers."""

    def __init__(self, win_len, subset=None):
        super().__init__(name="Synthetic", file_name='')  # We do not need to load data from a file
        self.win_len = win_len
        self.subset = subset

    def load(self):
        train = pd.read_csv('dataset/synthetic/noisy_sine_100_train.csv', header=None)
        test = pd.read_csv('dataset/synthetic/noisy_sine_100_test.csv', header=None)
        test_label = pd.read_csv('dataset/synthetic/noisy_sine_100_test_label.csv', header=None)

        x_train = train.values
        y_train = np.array([0 for _ in range(x_train.size)])
        x_test = test.values
        y_test = np.array([1 if x > 0 else 0 for x in pd.Series(test_label.values.ravel())
                          .groupby(np.arange(test_label.size) // self.win_len).sum()])

        y_test_binary = y_test

        self._data = tuple(pd.DataFrame(data=data) for data in [x_train, y_train, x_test, y_test, y_test_binary])
