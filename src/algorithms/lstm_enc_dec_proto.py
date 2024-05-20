import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from tqdm import trange
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from .algorithm_utils import Algorithm, PyTorchUtils

from src.algorithms.dummy_ae import LSTMAutoencoder

class LSTMEDProto(Algorithm, PyTorchUtils):
    def __init__(self, name: str = 'LSTM-ED', num_epochs: int = 10, batch_size: int = 20, lr: float = 1e-3,
                 hidden_size: int = 5, sequence_length: int = 30, step_size: int = 1,
                 train_gaussian_percentage: float = 0.25,
                 n_layers: tuple = (1, 1), n_proto: int = 2, dmin: int = 1, use_bias: tuple = (True, True),
                 dropout: tuple = (0, 0), bidirectional: bool = False,
                 lambda_c: float = None, lambda_e: float = None, lambda_d: float = None,
                 lambda_r: float = None, seed: int = None, proto_mapping: bool=False, gpu: int = None, details=True):
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

        self.n_proto = n_proto
        self.lambda_c = lambda_c
        self.lambda_e = lambda_e
        self.lambda_d = lambda_d
        self.dmin = dmin
        self.beta = lambda_r
        self.proto_mapping = proto_mapping

        self.gpu = gpu
        self.lstmed = None
        self.mean, self.cov = None, None

    def fit(self, X: pd.DataFrame):
        print("Training...")
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in
                     range(0, data.shape[0] - self.sequence_length + 1, self.step_size)]
        indices = np.random.permutation(len(sequences))
        split_point = int(self.train_gaussian_percentage * len(sequences))
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=False,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        val_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=False,
                                sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        self.lstmed = LSTMAutoencoder(self.sequence_length, X.shape[1], self.hidden_size, n_proto=self.n_proto, is_training=True, gpu=self.gpu)
        self.to_device(self.lstmed)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)
        train_loss = dict()
        val_loss = dict()
        train_d_loss = dict()
        train_c_loss = dict()
        train_e_loss = dict()
        train_r_loss = dict()

        train_embedding_buf = []
        train_raw_buf = []
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch + 1}/{self.num_epochs}.')
            train_loss_tmp = []
            val_loss_tmp = []
            loss_d_tmp = []
            loss_e_tmp = []
            loss_c_tmp = []
            loss_r_tmp = []
            train_embedding_buf = []
            train_raw_buf = []
            self.lstmed.train()
            for ts_batch in train_loader:
                output, embedding = self.lstmed(self.to_var(ts_batch))
                train_embedding_buf.append(embedding)
                train_raw_buf.append(ts_batch)
                loss_recon = nn.MSELoss(reduction='mean')(output, self.to_var(ts_batch.float()))
                loss_c, loss_e, loss_d = self._proto_loss(self.lstmed.prototypes, embedding, self.lambda_c,
                                                          self.lambda_e,
                                                          self.lambda_d) if self.n_proto != 0 else (0, 0, 0)
                loss = loss_c + loss_e + loss_d + self.beta * loss_recon if self.n_proto != 0 else loss_recon
                if self.n_proto != 0:
                    loss_d_tmp.append(loss_d.item())
                    loss_c_tmp.append(loss_c.item())
                    loss_e_tmp.append(loss_e.item())
                    loss_r_tmp.append(self.beta * loss_recon.item())

                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_tmp.append(loss.item())

            self.lstmed.eval()
            for ts_batch in val_loader:
                output, embedding = self.lstmed(self.to_var(ts_batch))
                loss_recon = nn.MSELoss(reduction='mean')(output, self.to_var(ts_batch.float()))
                v_loss_c, v_loss_e, v_loss_d = self._proto_loss(self.lstmed.prototypes, embedding, self.lambda_c,
                                                                self.lambda_e,
                                                                self.lambda_d) if self.n_proto != 0 else (0, 0, 0)
                loss = v_loss_c + v_loss_e + v_loss_d + self.beta * loss_recon if self.n_proto != 0 else loss_recon
                val_loss_tmp.append(loss.item())
            train_loss[epoch] = sum(train_loss_tmp) / len(train_loader.dataset)
            val_loss[epoch] = sum(val_loss_tmp) / len(val_loader.dataset)
            if self.n_proto != 0:
                train_d_loss[epoch] = sum(loss_d_tmp) / len(train_loader.dataset)
                train_c_loss[epoch] = sum(loss_c_tmp) / len(train_loader.dataset)
                train_e_loss[epoch] = sum(loss_e_tmp) / len(train_loader.dataset)
                train_r_loss[epoch] = sum(loss_r_tmp) / len(train_loader.dataset)

            if self.proto_mapping and epoch % 4 == 0 and self.n_proto != 0 and epoch > 1:
                self.lstmed.prototypes = self.map_prototypes_to_neighbors(train_embedding_buf)

        final_train_embeddings = torch.vstack(train_embedding_buf)
        decoded_protos = self._decode_protos(self.lstmed.prototypes, final_train_embeddings, train_raw_buf) \
            if self.n_proto != 0 else torch.tensor([[]])

        self.lstmed.eval()
        error_vectors = []
        for ts_batch in train_loader:
            output, _ = self.lstmed(self.to_var(ts_batch))
            error = nn.L1Loss(reduction='none')(output, self.to_var(ts_batch.float()))
            error_vectors += error.view(-1, X.shape[1])
        errors = torch.vstack(error_vectors)

        self.mean = torch.mean(errors, axis=0)
        self.cov = torch.cov(errors.T) if errors.shape[-1] > 1 else torch.var(errors)

        return train_loss, val_loss, self.lstmed.prototypes.data.cpu().numpy(), decoded_protos.data.cpu().numpy(), \
               self.mean.data.cpu().numpy(), self.cov.data.cpu().numpy(), (
               train_d_loss, train_c_loss, train_e_loss, train_r_loss)

    def map_prototypes_to_neighbors(self, train_embeddding_buf):
        embeddings = torch.vstack(train_embeddding_buf)
        dist = torch.cdist(embeddings, self.lstmed.prototypes)
        dist_idx = dist.sort(axis=0, descending=False).indices

        target = [-1 for _ in range(self.lstmed.prototypes.shape[0])]
        for i, row in enumerate(dist_idx):
            for num, neigh_idx in enumerate(row):
                if target[num] == -1:
                    if neigh_idx not in target:
                        target[num] = neigh_idx
                    else:
                        continue
        f = torch.tensor(target).ravel()
        f[f == 1] = random.randint(0, neigh_idx)
        return Parameter(embeddings[f])

    def _proto_loss(self, prototypes, embeddings, lambda_c, lambda_e, lambda_d):
        dist = torch.cdist(prototypes, embeddings)
        dist_pp = torch.cdist(prototypes, prototypes)

        # 1. average distance from points in the batch to the nearest prototypes
        min_values = dist.min(axis=1).values
        d_c = min_values.mean()

        # 2. average distance from every prototype to the nearest real data point
        min_values = dist.min(axis=0).values
        d_e = min_values.mean()

        # 3. distance between prototypes
        tmp = self.dmin * 20 - dist_pp
        tmp[tmp < 0] = 0
        d_d = tmp.sum() / 2

        return d_c * lambda_c, d_e * lambda_e, d_d * lambda_d

    def _decode_protos(self, prototypes, train_embeddings, train_raw_buf):
        dist = torch.cdist(prototypes, train_embeddings)
        idx_of_nearest_neighbors = dist.min(axis=1).indices
        decoded_protos = torch.vstack(train_raw_buf)[idx_of_nearest_neighbors]
        return decoded_protos.reshape(-1, decoded_protos.shape[-1])

    def predict(self, X: pd.DataFrame):
        print("Predicting...")
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in
                     range(0, data.shape[0] - self.sequence_length + 1, self.step_size)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)
        m_normal = MultivariateNormal(self.mean, self.cov) if X.shape[-1] > 1 else Normal(self.mean, self.cov)
        self.lstmed.eval()
        outputs = []
        errors = []
        scores = []
        embeddings = []
        for idx, ts in enumerate(data_loader):
            output, hidden = self.lstmed(self.to_var(ts))
            error = nn.L1Loss(reduction='none')(output, self.to_var(ts.float()))
            score = self._anomaly_score(error, m_normal, method='log_pdf')

            outputs.append(output)
            errors.append(error)
            embeddings.append(hidden)
            scores.append(score)

        outputs = torch.vstack(outputs)  # shape: N * win_len * dim
        errors = torch.vstack(errors)  # shape: N * win_len * dim
        scores = pd.concat(scores, axis=0)
        embeddings = torch.vstack(embeddings)

        outputs = pd.DataFrame(outputs.data.cpu().numpy().reshape(-1, outputs.shape[-1]))
        errors = pd.DataFrame(errors.data.cpu().numpy().reshape(-1, errors.shape[-1]))
        embeddings = pd.DataFrame(embeddings.data.cpu().numpy().reshape(-1, self.hidden_size))

        return scores, outputs, errors, embeddings

    def _anomaly_score(self, reconstruction_error, m_normal, method):
        error = reconstruction_error.view(-1, reconstruction_error.shape[-1])
        if method == 'max_win_dim':
            error = pd.DataFrame(reconstruction_error)
            scores = error.groupby(np.arange(error.shape[0]) // self.sequence_length).max().max(axis=1)
        elif method == 'avg_win_dim':
            error = pd.DataFrame(reconstruction_error)
            scores = error.groupby(np.arange(error.shape[0]) // self.sequence_length).avg().avg(axis=1)
        elif method == 'm_dist':
            scores = torch.diag((error - self.mean).matmul(self.cov).matmul((error - self.mean).T)) if error.shape[
                                                                                                           -1] > 1 else self._pdf(
                error.ravel())
            scores = pd.DataFrame(scores.data.cpu().numpy()) \
                .groupby(np.arange(error.shape[0]) // self.sequence_length).max()
        elif method == 'log_pdf':
            scores = self._logpdf(error, m_normal)
            scores = pd.DataFrame(scores.data.cpu().numpy()) \
                .groupby(np.arange(error.shape[0]) // self.sequence_length).max()
        else:
            print(f'Unsupported score calculation method {method}')
        return scores

    def _pdf(self, error):
        result = []
        for err in error:
            result.append((1 / (self.cov * torch.sqrt(torch.tensor(2 * np.pi)))) \
                          * np.e ** (-(err - self.mean) ** 2 / (2 * self.cov ** 2)))
        return torch.vstack(result)

    def _logpdf(self, error, m_normal):
        return -m_normal.log_prob(error)

    def evaluation(self, label, scores, anomaly_rate=0.1):
        '''
        label for windows, predictions for points
        '''
        print("Evaluating...")
        label = label[:scores.shape[0]]
        threshold = np.quantile(scores, 1 - anomaly_rate)
        binary_pred = pd.Series([0 if score < threshold else 1 for score in scores.values])
        tn, fp, fn, tp = confusion_matrix(label, binary_pred).ravel()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall)
        fpr, tpr, thresholds = metrics.roc_curve(label, scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy, 'fpr': str(fpr),
                'tpr': str(tpr), 'thresholds': str(tpr), 'auc': auc}


class LSTMEDProtoModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int, n_proto: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple, bidirectional: bool,
                 seed: int, gpu: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.num_direction = 2 if bidirectional else 1

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0],
                               dropout=self.dropout[0], bidirectional=bidirectional)

        prototypes = torch.Tensor(n_proto, hidden_size)
        self.prototypes = Parameter(prototypes)
        nn.init.uniform_(self.prototypes, -1, 1)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1],
                               dropout=self.dropout[1], bidirectional=bidirectional)

        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)

    def _init_hidden(self, batch_size):
        return (self.to_var(torch.Tensor(self.n_layers[0] * self.num_direction, batch_size, self.hidden_size).zero_()),
                self.to_var(torch.Tensor(self.n_layers[0] * self.num_direction, batch_size, self.hidden_size).zero_()))

    def forward(self, ts_batch, return_latent: bool = True):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        encoder_output, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)

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
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return (output, enc_hidden[1][-1]) if return_latent else output
