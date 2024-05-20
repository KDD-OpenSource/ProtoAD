import warnings

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import json
import os
import itertools
import time
from datetime import datetime
import matplotlib.pyplot as plt
from src.algorithms import LSTMED, LSTMEDProto, DAGMM
from src.datasets import ECG5000, SMD, Taxi, SMAP, MSL, Synthetic, Arrhythmia

datasets = ['SMD#1-2', 'SMD#1-3', 'SMD#2-1', 'SMD#2-2', 'SMD#2-3', 'SMD#3-1', 'SMD#3-2', 'SMD#3-3']
datasets = ['SMAP', 'Taxi', 'Synthetic', 'SMD#1-1']
datasets = ['Arrhythmia']
datasets += ['SMD#1-1', 'SMD#1-5', 'SMD#1-7', 'SMD#1-6', 'SMD#3-5', 'SMD#2-6', 'SMD#1-4', 'SMD#3-9', 'SMD#2-1',
             'SMD#3-8', 'SMD#3-11', 'SMD#3-6', 'SMD#2-4', 'SMD#3-3', 'SMD#2-8',
             'SMD#1-2', 'SMD#3-10', 'SMD#2-2', 'SMD#2-3', 'SMD#3-7', 'SMD#1-3', 'SMD#2-7', 'SMD#1-8', 'SMD#2-9',
             'SMD#3-2', 'SMD#3-4', 'SMD#2-5', 'SMD#3-1']
dataset_obj = {'ECG5000': ECG5000, 'SMD': SMD, 'Taxi': Taxi, 'SMAP': SMAP, 'MSL': MSL, 'Synthetic': Synthetic,
               'Arrhythmia': Arrhythmia}
gpus = [1]
seeds = [24]
output_folder = 'outputs/exp'

models = ['proto_ad', 'lstm_ed']

proto_mapping = True

for dataset in datasets:
    now = datetime.now()
    output_dir = f'{output_folder}/{dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset_name, subset_name = dataset.split('#')[0], dataset.split('#')[1] if len(dataset.split('#')) > 1 else None
    with open('configuration/config.json', 'r') as f:
        config = json.load(f)
    config_exp = config[dataset_name]

    n_protos = config_exp['n_protos']
    win_len = config_exp['win_len']
    hds = config_exp['hidden_size']
    epochs = config_exp['epoch']
    lrs = config_exp['lr']
    dropouts = config_exp['dropout']
    anomaly_rate = config_exp['anomaly_rate']
    batch_size = config_exp['batch_size']
    bidirs = config_exp['bidirectional']
    n_layers = config_exp['num_layers']
    lambda_c = config_exp['lambda_c']
    lambda_e = config_exp['lambda_e']
    lambda_d = config_exp['lambda_d']
    lambda_r = config_exp['lambda_r']
    dmin = config_exp['dmin'] if 'dmin' in config_exp else None

    x_train, y_train, x_test, y_test, y_test_binary = dataset_obj[dataset_name](win_len=win_len,
                                                                                subset=subset_name).data()

    for (model_name, hd, epoch, lr, dr, bidir, n_layer, n_proto, seed) in itertools.product(models, hds, epochs, lrs,
                                                                                            dropouts,
                                                                                            bidirs, n_layers, n_protos,
                                                                                            seeds):
        print(f'{dataset}: hd={hd}, epoch={epoch}, lr={lr}, dropout={dr}, nProto={n_proto}, seed={seed}.')
        train_start = time.time()
        num_layers = (n_layer, n_layer)  # for encode and decoder
        bidirectional = True if bidir == 1 else False
        identifier = f'hd{hd}_epoch{epoch}_lr{lr}_drop{dr}_bidir{bidir}_nlayer{n_layer}' if model_name != 'proto_ad' else f'hd{hd}_epoch{epoch}_lr{lr}_drop{dr}_bidir{bidir}_nlayer{n_layer}_nproto{n_proto}_lc{lambda_c}_ld{lambda_d}_le{lambda_e}_lambdar{lambda_r}_dmin{dmin}_seed_{seed}'

        if os.path.exists(f'{output_dir}/{identifier}/loss.csv'):
            print(f'Skip {dataset}/{identifier}')
            continue

        if model_name == 'proto_ad':
            model = LSTMEDProto(sequence_length=win_len, step_size=win_len, num_epochs=epoch, batch_size=batch_size,
                                hidden_size=hd, dropout=(dr, dr), bidirectional=bidirectional, n_layers=num_layers,
                                n_proto=n_proto, lr=lr, lambda_c=lambda_c, lambda_e=lambda_e, lambda_d=lambda_d,
                                lambda_r=lambda_r, dmin=dmin, proto_mapping=proto_mapping, gpu=gpus[0])
            train_loss, val_loss, prototypes, decoded_protos, mean, cov, tmp = model.fit(x_train)
            train_d_loss, train_c_loss, train_e_loss, train_r_loss = tmp
        elif model_name == 'lstm_ed':
            model = LSTMED(sequence_length=win_len, step_size=win_len, num_epochs=epoch, hidden_size=hd,
                           dropout=(dr, dr), bidirectional=bidirectional, n_layers=num_layers,
                           lr=lr, gpu=gpus[0])
            train_loss, val_loss = model.fit(x_train)
        else:
            warnings.warn('Model not implemented.')
        train_time = (time.time() - train_start) / epoch

        scores, outputs, errors, embeddings = model.predict(x_test)

        metrics = model.evaluation(y_test_binary, scores, anomaly_rate)

        os.mkdir(f'{output_dir}/{identifier}')
        pd.DataFrame(x_test).to_csv(f'{output_dir}/x_test.csv', header=None, index=None)
        pd.DataFrame(y_test).to_csv(f'{output_dir}/y_test.csv', header=None, index=None)
        pd.DataFrame(y_test_binary).to_csv(f'{output_dir}/y_test_binary.csv', header=None, index=None)
        pd.DataFrame(outputs).to_csv(f'{output_dir}/{identifier}/outputs.csv', header=None, index=None)
        pd.DataFrame(errors).to_csv(f'{output_dir}/{identifier}/errors.csv', header=None, index=None)
        pd.DataFrame(embeddings).to_csv(f'{output_dir}/{identifier}/embeddings.csv', header=None, index=None)
        pd.DataFrame(decoded_protos).to_csv(f'{output_dir}/{identifier}/decoded_protos.csv', header=None, index=None)
        pd.DataFrame(scores).to_csv(f'{output_dir}/{identifier}/scores.csv', header=None, index=None)

        if len(n_protos) != 0:
            pd.DataFrame(prototypes).to_csv(f'{output_dir}/{identifier}/prototypes.csv', header=None, index=None)
        with open(f'{output_dir}/{identifier}/mean.npy', 'wb') as f_m:
            np.save(f_m, mean)
        with open(f'{output_dir}/{identifier}/cov.npy', 'wb') as f_c:
            np.save(f_c, cov)

        log = {'dataset': dataset, 'window length': win_len, 'hidden size': hd, 'epoch': epoch, 'learning rate': lr,
               'dropout rate': dr, 'num prototypes': n_proto, 'seed': seed, 'training_time': train_time}
        with open(f'{output_dir}/{identifier}/config.json', 'w') as log_file:
            json.dump(log, log_file)
        with open(f'{output_dir}/{identifier}/metrics.json', 'w') as metrics_file:
            json.dump(metrics, metrics_file)

        ax = pd.Series(train_loss).plot(c='blue')
        pd.Series(val_loss).plot(c='orange', ax=ax)
        fig = ax.get_figure()
        fig.savefig(f'{output_dir}/{identifier}/loss.png')
        plt.cla()
        pd.concat([pd.Series(train_loss), pd.Series(val_loss)], axis=1).to_csv(f'{output_dir}/{identifier}/loss.csv',
                                                                               header=None, index=None)
        exp_time = (time.time() - train_start)
        print(f'Time used for experiment: {exp_time}')

        pd.concat([pd.Series(train_d_loss), pd.Series(train_c_loss), pd.Series(train_e_loss), pd.Series(train_r_loss)],
                  axis=1).to_csv(f'{output_dir}/{identifier}/loss_detail.csv', header=None, index=None)

        ax1 = pd.Series(train_d_loss).plot(c='blue')
        fig = ax1.get_figure()
        fig.savefig(f'{output_dir}/{identifier}/loss_d.png')
        plt.cla()

        ax1 = pd.Series(train_c_loss).plot(c='blue')
        fig = ax1.get_figure()
        fig.savefig(f'{output_dir}/{identifier}/loss_c.png')
        plt.cla()

        ax1 = pd.Series(train_e_loss).plot(c='blue')
        fig = ax1.get_figure()
        fig.savefig(f'{output_dir}/{identifier}/loss_e.png')
        plt.cla()

        ax1 = pd.Series(train_r_loss).plot(c='blue')
        fig = ax1.get_figure()
        fig.savefig(f'{output_dir}/{identifier}/loss_r.png')
        plt.cla()
