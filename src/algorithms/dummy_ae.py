import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.parameter import Parameter

class LSTMAutoencoder(nn.Module):

    def __init__(self, seq_len, n_features, hidden_size, n_proto, is_training, bidirectional=False, n_layers=(1, 1), gpu=None):
        super(LSTMAutoencoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_direction = 2 if bidirectional else 1
        self.n_layers = n_layers

        self.encoder = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], dropout=0.2, bidirectional=self.bidirectional)

        self.decoder = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], dropout=0.2, bidirectional=self.bidirectional)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.is_training = is_training
        self.device = f'cuda:{gpu}'
        prototypes = torch.Tensor(n_proto, self.hidden_size)
        self.prototypes = Parameter(prototypes)
        nn.init.uniform_(self.prototypes, -1, 1)

    def _init_hidden(self, batch_size, device):
        return torch.Tensor(self.n_layers[0] * self.num_direction, batch_size, self.hidden_size).zero_().to(
            f'cuda:{device}'), torch.Tensor(self.n_layers[0] * self.num_direction, batch_size,
                                            self.hidden_size).zero_().to(f'cuda:{device}')

    def to_var(self, t, **kwargs):
        t = t.to(self.device)
        return Variable(t, **kwargs)

    def forward(self, ts_batch):
        batch_size = ts_batch.shape[0]
        device = ts_batch.get_device()
        init_enc_hidden = self._init_hidden(batch_size, device)
        _, enc_hidden = self.encoder(ts_batch.float(), init_enc_hidden)
        dec_hidden = enc_hidden
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])
            if self.is_training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return output,  enc_hidden[1][-1]
