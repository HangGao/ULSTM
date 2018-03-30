import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

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

class FPLSTM_Layer(nn.Module):
    r"""Applies a single layer Full-Peephole LSTM layer to an input sequence. Only support 1 layer in our case
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
        super(FPLSTM_Layer, self).__init__()
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
        else:
            hidden_h, hidden_c = hidden
            hidden_h = torch.squeeze(hidden_h)
            hidden_c = torch.squeeze(hidden_c)

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

            u = iozfux[:, 4 * self.hidden_size:] + self.um(torch.mul(z, F.tanh(hidden_c)))
            u = F.tanh(u)

            hidden_c = torch.mul(i, u) + torch.mul(f, hidden_c)
            hidden_h = torch.mul(o, F.tanh(hidden_c))

            # (1, batch_size, hidden_size)
            output.append(torch.unsqueeze(hidden_h, 0))

        output = torch.cat(output, 0)
        hidden_c = torch.unsqueeze(hidden_c, 0)
        hidden_h = torch.unsqueeze(hidden_h, 0)
        return output, (hidden_h, hidden_c)
