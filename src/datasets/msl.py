import pandas as pd
from .nasa_helper import load_nasa
from .dataset import Dataset


class MSL(Dataset):

    def __init__(self, win_len, subset=None):
        super().__init__(name="MSL", file_name='')
        self.win_len = win_len
        self.subset = subset

    def load(self):
        x_train, y_train, x_test, y_test, y_test_binary = load_nasa('dataset/nasa', 'MSL', self.win_len)
        self._data = tuple(pd.DataFrame(data=data) for data in [x_train, y_train, x_test, y_test, y_test_binary])

