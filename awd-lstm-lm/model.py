import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class CMul(nn.Module):
    def __init__(self, in_dim):
        super(CMul, self).__init__()
        self.in_dim = in_dim
        self.weight = nn.Parameter(torch.Tensor(in_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_dim)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return torch.mul(self.weight.repeat(input.size(0), 1), input)

class ULSTM_Layer(nn.Module):
    r"""Applies a single layer Untied LSTM layer to an input sequence. Only support 1 layer in our case
    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (batch, hidden_size): tensor containing the initial hidden state for the QRNN.
    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size=None):
        super(ULSTM_Layer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if self.hidden_size is None:
            self.hidden_size = input_size

        self.iozfux = nn.Linear(self.input_size, 5 * self.hidden_size)
        self.iozfh = nn.Linear(self.hidden_size, 4 * self.hidden_size)
        self.um = nn.Linear(self.hidden_size, self.hidden_size)

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()

        if hidden is None:
            # size (batch_size, hidden_size)
            hidden_h = Variable(X.data.new(batch_size, self.hidden_size).fill_(0.))
            hidden_c = Variable(X.data.new(batch_size, self.hidden_size).fill_(0.))
            hidden_c_tanh = F.tanh(hidden_c)
        else:
            hidden_h, hidden_c = hidden
            hidden_h = torch.squeeze(hidden_h)
            hidden_c = torch.squeeze(hidden_c)
            hidden_c_tanh = F.tanh(hidden_c)

        output = []
        for i in range(seq_len):
            x = X[i]       # size (batch_size, input_size)
            # (batch_size, 5 * hidden_size)
            iozfux = self.iozfux(x)
            # (batch_size, 4 * hidden_size)
            iozfh = self.iozfh(hidden_h)
            # (batch_size, 4 * hidden_size)
            iozf = F.sigmoid(iozfux[:, :4 * self.hidden_size] + iozfh)
            i, o, z, f = torch.split(iozf, iozf.size(1) // 4, dim=1)

            u = iozfux[:, 4 * self.hidden_size:] + self.um(torch.mul(z, hidden_c_tanh))
            u = F.tanh(u)

            hidden_c = torch.mul(i, u) + torch.mul(f, hidden_c)
            hidden_c_tanh = F.tanh(hidden_c)
            hidden_h = torch.mul(o, hidden_c_tanh)

            # (1, batch_size, hidden_size)
            output.append(torch.unsqueeze(hidden_h, 0))

        output = torch.cat(output, 0)
        hidden_c = torch.unsqueeze(hidden_c, 0)
        hidden_h = torch.unsqueeze(hidden_h, 0)
        return output, (hidden_h, hidden_c)

class PLSTM_Layer(nn.Module):
    r"""Applies a single LSTM layer with peephole connections to an input sequence. Only support 1 layer in our case
    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (batch, hidden_size): tensor containing the initial hidden state for the QRNN.
    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size=None):
        super(PLSTM_Layer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if self.hidden_size is None:
            self.hidden_size = input_size

        self.iozfx = nn.Linear(self.input_size, 4 * self.hidden_size)
        self.iozfh = nn.Linear(self.hidden_size, 4 * self.hidden_size)

        self.zc = nn.Parameter(torch.Tensor(1, hidden_size))
        stdv = 1.0 / math.sqrt(hidden_size)
        self.zc.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()

        if hidden is None:
            # size (batch_size, hidden_size)
            hidden_h = Variable(X.data.new(batch_size, self.hidden_size).fill_(0.))
            hidden_c = Variable(X.data.new(batch_size, self.hidden_size).fill_(0.))
        else:
            hidden_h, hidden_c = hidden
            hidden_h = torch.squeeze(hidden_h)
            hidden_c = torch.squeeze(hidden_c)

        output = []
        for i in range(seq_len):
            x = X[i]       # size (batch_size, input_size)
            iozf = self.iozfx(x) + self.iozfh(hidden_h)
            i, o, z, f = torch.split(iozf, iozf.size(1) // 4, dim=1)
            i, o, f = F.sigmoid(i), F.sigmoid(o), F.sigmoid(f)
            try:
                u = torch.mul(self.zc.expand(hidden_c.size()), hidden_c)
            except:
                print(self.zc.size())
                print(hidden_c.size())
            z = F.tanh(z + u)

            hidden_c = torch.mul(i, z) + torch.mul(f, hidden_c)
            hidden_c_tanh = F.tanh(hidden_c)
            hidden_h = torch.mul(o, hidden_c_tanh)

            # (1, batch_size, hidden_size)
            output.append(torch.unsqueeze(hidden_h, 0))

        output = torch.cat(output, 0)
        hidden_c = torch.unsqueeze(hidden_c, 0)
        hidden_h = torch.unsqueeze(hidden_h, 0)
        return output, (hidden_h, hidden_c)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and with/without a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, use_decoder=True):
        super(RNNModel, self).__init__()
        self.use_decoder = use_decoder
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU', 'ULSTM', 'PLSTM'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'ULSTM':
            self.rnns = [ULSTM_Layer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid)) for l in range(nlayers)]
            if wdrop:
                for rnn in self.rnns:
                    rnn.iozfh = WeightDrop(rnn.iozfh, ['weight'], dropout=wdrop)
                    rnn.um = WeightDrop(rnn.um, ['weight'], dropout=wdrop)
        elif rnn_type == 'PLSTM':
            self.rnns = [PLSTM_Layer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid)) for l in range(nlayers)]
            if wdrop:
                for rnn in self.rnns:
                    rnn.iozfh = WeightDrop(rnn.iozfh, ['weight'], dropout=wdrop)
        elif rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        if use_decoder:
            self.decoder = nn.Linear(nhid, ntoken)

            # Optionally tie weights as in:
            # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
            # https://arxiv.org/abs/1608.05859
            # and
            # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
            # https://arxiv.org/abs/1611.01462
            if tie_weights:
                #if nhid != ninp:
                #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
                self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.use_decoder:
            self.decoder.bias.data.fill_(0)
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM' or self.rnn_type == 'ULSTM' or self.rnn_type == 'PLSTM':
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()),
                    Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
