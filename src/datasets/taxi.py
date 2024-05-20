# Adapted from https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection/blob/master/0_download_dataset.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .dataset import Dataset


class Taxi(Dataset):

    def __init__(self, win_len, subset=None):
        super().__init__(name="Taxi", file_name='')
        self.win_len = win_len
        self.subset = subset

    def load(self):
        df = pd.read_csv('dataset/taxi/nyc_taxi.csv', header=None)

        label = [1 if 150 < i < 250 or \
                      5970 < i < 6050 or \
                      8500 < i < 8650 or \
                      8750 < i < 8890 or \
                      10000 < i < 10200 or \
                      14700 < i < 14800 \
                     else 0
                 for i in range(df.shape[0])]
        new_df = pd.concat([df.iloc[:,1], pd.Series(label)], axis=1)

        # sampling rate is 30 min, so 48 records are from one day.
        train_buf = []
        test_buf = []

        for i in range(new_df.shape[0] // 48):
            if i <= (new_df.shape[0] // 48) * 0.7:
                if sum(new_df.iloc[i * 48:(i + 1) * 48, 1]) == 0:
                    train_buf.append(new_df.iloc[i * 48:(i + 1) * 48, :])
                else:
                    test_buf.append(new_df.iloc[i * 48:(i + 1) * 48, :])
            else:
                test_buf.append(new_df.iloc[i * 48:(i + 1) * 48, :])

        x_train = pd.concat(train_buf, axis=0).iloc[:, :-1].values.ravel()
        y_train = pd.concat(train_buf, axis=0).iloc[:, -1].values

        x_test = pd.concat(test_buf, axis=0).iloc[:, :-1].values.ravel()
        y_test = pd.concat(test_buf, axis=0).iloc[:, -1].values

        y_test_binary = y_test

        train_scaler = MinMaxScaler().fit(x_train.reshape(-1, 1))
        x_train_scaled = train_scaler.transform(x_train.reshape(-1, 1))
        test_scaler = MinMaxScaler().fit(x_test.reshape(-1, 1))
        x_test_scaled = test_scaler.transform(x_test.reshape(-1, 1))

        self._data = tuple(pd.DataFrame(data=data) for data in [x_train_scaled, y_train, x_test_scaled, y_test, y_test_binary])
