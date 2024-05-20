import pandas as pd
import numpy as np
from .dataset import Dataset


class SMD(Dataset):
    """
    First machine data in the SMA dataset, 0 is normal.
    """

    def __init__(self, win_len, subset=None):
        super().__init__(name="SMD", file_name='')
        self.win_len = win_len
        self.subset = subset

    def load(self):
        train = pd.read_csv(f'dataset/smd/{self.subset}/train.csv', header=None)
        test = pd.read_csv(f'dataset/smd/{self.subset}/test.csv', header=None)
        test_label = pd.read_csv(f'dataset/smd/{self.subset}/test_label.csv', header=None)

        x_train = train.values
        y_train = pd.Series([0 for _ in range(x_train.shape[0])]).values
        x_test = test.values
        y_test = np.array([1 if x > 0 else 0 for x in pd.Series(test_label.values.ravel())
                          .groupby(np.arange(test_label.size) // self.win_len).sum()])
        y_test_binary = y_test

        self._data = tuple(pd.DataFrame(data=data) for data in [x_train, y_train, x_test, y_test, y_test_binary])
