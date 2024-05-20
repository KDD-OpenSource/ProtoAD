import pandas as pd
from .dataset import Dataset
from .nasa_helper import load_nasa


class SMAP(Dataset):

    def __init__(self, win_len, subset=None):
        super().__init__(name="SMAP", file_name='')
        self.win_len = win_len
        self.subset = subset

    def load(self):
        x_train, y_train, x_test, y_test, y_test_binary = load_nasa('dataset/nasa', 'SMAP', self.win_len)
        self._data = tuple(pd.DataFrame(data=data) for data in [x_train, y_train, x_test, y_test, y_test_binary])


