# adapted from: https://github.com/KDD-OpenSource/DeepADoTS/blob/master/src/algorithms/lstm_enc_dec_axl.py
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

from .algorithm_utils import Algorithm, PyTorchUtils

class LSTMED(Algorithm, PyTorchUtils):
    def __init__(self, name: str = 'LSTM-ED', num_epochs: int = 10, batch_size: int = 20, lr: float = 1e-3,
                 hidden_size: int = 5, sequence_length: int = 30, step_size: int = 1,
                 train_gaussian_percentage: float = 0.25,
                 n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0),
                 bidirectional: bool = False, seed: int = None, gpu: int = None, details=True):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstmed = None
        self.mean, self.cov = None, None

    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in
                     range(0, data.shape[0] - self.sequence_length + 1, self.step_size)]
        indices = np.random.permutation(len(sequences))
        split_point = int(self.train_gaussian_percentage * len(sequences))
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        self.lstmed = LSTMEDModule(X.shape[1], self.hidden_size,
                                   self.n_layers, self.use_bias, self.dropout, self.bidirectional,
                                   seed=self.seed, gpu=self.gpu)
        self.to_device(self.lstmed)

        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)
        train_loss = dict()
        val_loss = dict()
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch + 1}/{self.num_epochs}.')
            train_loss_tmp = []
            val_loss_tmp = []
            self.lstmed.train()
            for ts_batch in train_loader:
                output, _ = self.lstmed(self.to_var(ts_batch))
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_tmp.append(loss.item())

            self.lstmed.eval()
            for ts_batch in train_gaussian_loader:
                output, _ = self.lstmed(self.to_var(ts_batch))
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                val_loss_tmp.append(loss.item())
            train_loss[epoch] = sum(train_loss_tmp) / len(train_loader.dataset)
            val_loss[epoch] = sum(val_loss_tmp) / len(train_gaussian_loader.dataset)

        return train_loss, val_loss

    def predict(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in
                     range(0, data.shape[0] - self.sequence_length + 1, self.step_size)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.lstmed.eval()
        outputs = []
        errors = []
        embeddings = []
        for idx, ts in enumerate(data_loader):

            output, hidden = self.lstmed(self.to_var(ts))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            if self.details:
                outputs.append(output.data.cpu().numpy())
                errors.append(error.data.cpu().numpy())
                embeddings.append(hidden.data.cpu().numpy())

        if self.details:
            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = output
            self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = error
            self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})

            embeddings = np.concatenate(embeddings)

        scores = self._anomaly_score(outputs, method='max_win_dim')

        return scores, pd.DataFrame(outputs.reshape(-1, outputs.shape[-1])), pd.DataFrame(embeddings)

    def _anomaly_score(self, reconstruction_error, method):
        error = pd.DataFrame(reconstruction_error)
        if method == 'max_win_dim':
            scores = error.groupby(np.arange(error.shape[0]) // self.sequence_length).max()
        elif method == 'avg_win_dim':
            scores = error.groupby(np.arange(error.shape[0]) // self.sequence_length).avg()
        # elif method == 'm_dist':
        #     scores = np.diag(reconstruction_error.dot(self.cov).dot(reconstruction_error.T))
        else:
            print(f'Unsupported score calculation method {method}')

        return scores

    def evaluation(self, label, scores):
        '''
        label for windows, predictions for points
        '''

        threshold = np.quantile(scores, 0.9) 

        binary_pred = pd.Series([0 if score < threshold else 1 for score in scores.values])

        precision, recall, f1, _ = precision_recall_fscore_support(label, binary_pred, average='micro')
        fpr, tpr, thresholds = metrics.roc_curve(label, binary_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return {'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fpr, 'tpr': tpr, 'thresholds': tpr,
                'auc': auc}


class LSTMEDModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple,
                 bidirectional: bool, seed: int, gpu: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.num_direction = 2 if bidirectional else 1


        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0],
                               bidirectional=bidirectional)
        self.to_device(self.encoder)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1],
                               bidirectional=bidirectional)
        self.to_device(self.decoder)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        return (self.to_var(torch.Tensor(self.n_layers[0]*self.num_direction, batch_size, self.hidden_size).zero_()),
                self.to_var(torch.Tensor(self.n_layers[0]*self.num_direction, batch_size, self.hidden_size).zero_()))

    def forward(self, ts_batch, return_latent: bool = True):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = enc_hidden

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                # _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)

        return (output, enc_hidden[1][-1]) if return_latent else output
