# adapted from: https://github.com/NetManAIOps/OmniAnomaly/blob/master/data_preprocess.py
import csv
import os
import ast
import numpy as np
import pandas as pd


def load_nasa(dataset_folder, dataset, win_len):
    with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        res = [row for row in csv_reader][1:]
    res = sorted(res, key=lambda k: k[0])
    # label_folder = os.path.join(dataset_folder, 'test_label')
    # os.makedirs(label_folder, exist_ok=True)
    data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
    labels = []
    for row in data_info:
        anomalies = ast.literal_eval(row[2])
        length = int(row[-1])
        label = np.zeros([length], dtype=np.bool)
        for anomaly in anomalies:
            label[anomaly[0]:anomaly[1] + 1] = True
        labels.extend(label)
    labels = np.asarray(labels)

    x_train = concatenate_and_save(dataset_folder, 'train', data_info)
    x_test = concatenate_and_save(dataset_folder, 'test', data_info)

    y_train = np.array([0 for _ in range(x_train.shape[0])])
    y_test = np.array([1 if x > 0 else 0 for x in pd.Series(labels)
                      .groupby(np.arange(labels.size) // win_len).sum()])
    y_test_binary = y_test

    return x_train, y_train, x_test, y_test, y_test_binary


def concatenate_and_save(dataset_folder, category, data_info):
    data = []
    for row in data_info:
        filename = row[0]
        temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
        data.extend(temp)
    data = np.asarray(data)

    return data
