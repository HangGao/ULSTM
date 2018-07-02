import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from torch.autograd import Function as Function
from functools import reduce
import math

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

class ULSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_o=True):
        super(ULSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.use_o = use_o

        self.iozfux = nn.Linear(self.in_dim, 5 * self.mem_dim)
        self.iozfh = nn.Linear(self.mem_dim, 4 * self.mem_dim)
        self.um = nn.Linear(self.mem_dim, self.mem_dim)

    def reset(self):
        # reset forget gate bias
        self.iozfux.bias.data[3*self.mem_dim : 4*self.mem_dim].fill_(1.5)
        self.iozfh.bias.data[3*self.mem_dim : 4*self.mem_dim].fill_(1.5)

    def node_forward(self, c_t_1, h_t_1, input):
        '''
        c_t_1   -->   (1, mem_dim)
        h_t_1   -->   (1, mem_dim)
        input   -->   (1, in_dim)

        '''
        iozfux = torch.unsqueeze(self.iozfux(input), 0)
        ix, ox, zx, fx, ux = torch.split(iozfux, iozfux.size(1) // 5, dim=1)
        iozfh = self.iozfh(h_t_1)
        ih, oh, zh, fh = torch.split(iozfh, iozfh.size(1) // 4, dim=1)

        """
        Transitions:
            i = sigmoid(W_i * input + U_i * h_t_1 + b_i)
            f = sigmoid(W_f * input + U_f * h_t_1 + b_f)
            o = sigmoid(w_o * input + U_o * h_t_1 + b_o)
            z = sigmoid(W_z * input + U_z * h_t_1 + b_z)
            candidate = tanh(W * input + M * (z \odot tanh(c_t_1)) + b)
            c_t = i \odot candidate + f \odot c_t_1
            h_t = o \odot tanh(c_t)
        """
        i, o, f, z = F.sigmoid(ix + ih), F.sigmoid(ox + oh), F.sigmoid(fx + fh), F.sigmoid(zx + zh)

        u = ux + self.um(torch.mul(z, F.tanh(c_t_1)))
        u = F.tanh(u)

        c = torch.mul(i, u) + torch.mul(f, c_t_1)

        if self.use_o:
            h = torch.mul(o, F.tanh(c))
        else:
            h = F.tanh(c)
        return c, h

    def forward(self, inputs):
        c_t, h_t = None, None
        for i in range(inputs.size(0)):
            if i == 0:
                c_t_1 = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
                h_t_1 = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            else:
                c_t_1 = c_t
                h_t_1 = h_t
            c_t, h_t = self.node_forward(c_t_1, h_t_1, inputs[i])
        return c_t, h_t

class PLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_o=True):
        super(PLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.use_o = use_o

        self.iofux = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.iofuh = nn.Linear(self.mem_dim, 4 * self.mem_dim)

        self.zc = nn.Parameter(torch.Tensor(1, self.mem_dim))
        stdv = 1.0 / math.sqrt(self.mem_dim)
        self.zc.data.uniform_(-stdv, stdv)


    def reset(self):
        # reset forget gate bias
        self.iofux.bias.data[2*self.mem_dim : 3*self.mem_dim].fill_(1.5)
        self.iofux.bias.data[2*self.mem_dim : 3*self.mem_dim].fill_(1.5)

    def node_forward(self, c_t_1, h_t_1, input):
        '''
        c_t_1   -->   (1, mem_dim)
        h_t_1   -->   (1, mem_dim)
        input   -->   (1, in_dim)
        '''
        iofu = self.iofux(input) + self.iofuh(h_t_1)
        i, o, f, u = torch.split(iofu, iofu.size(1) // 4, dim=1)

        """
        Transitions:
            i = sigmoid(W_i * input + U_i * h_t_1 + b_i)
            f = sigmoid(W_f * input + U_f * h_t_1 + b_f)
            o = sigmoid(w_o * input + U_o * h_t_1 + b_o)
            candidate = tanh(W * input + U * h_t_1 + b)
            c_t = i \odot candidate + f \odot c_t_1
            h_t = o \odot tanh(c_t)
        """
        i, o, f = F.sigmoid(i), F.sigmoid(o), F.sigmoid(f)
        u = F.tanh(u + torch.mul(self.zc.expand(c_t_1.size(0), self.mem_dim), c_t_1))

        c = torch.mul(i, u) + torch.mul(f, c_t_1)

        if self.use_o:
            h = torch.mul(o, F.tanh(c))
        else:
            h = F.tanh(c)
        return c, h

    def forward(self, inputs):
        c_t, h_t = None, None
        for i in range(inputs.size(0)):
            if i == 0:
                c_t_1 = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
                h_t_1 = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            else:
                c_t_1 = c_t
                h_t_1 = h_t
            c_t, h_t = self.node_forward(c_t_1, h_t_1, inputs[i])
        return c_t, h_t

class LSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_o=True):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.use_o = use_o

        self.iofux = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.iofuh = nn.Linear(self.mem_dim, 4 * self.mem_dim)

    def reset(self):
        # reset forget gate bias
        self.iofux.bias.data[2*self.mem_dim : 3*self.mem_dim].fill_(1.5)
        self.iofux.bias.data[2*self.mem_dim : 3*self.mem_dim].fill_(1.5)

    def node_forward(self, c_t_1, h_t_1, input):
        '''
        c_t_1   -->   (1, mem_dim)
        h_t_1   -->   (1, mem_dim)
        input   -->   (1, in_dim)
        '''
        iofu = self.iofux(input) + self.iofuh(h_t_1)
        i, o, f, u = torch.split(iofu, iofu.size(1) // 4, dim=1)

        """
        Transitions:
            i = sigmoid(W_i * input + U_i * h_t_1 + b_i)
            f = sigmoid(W_f * input + U_f * h_t_1 + b_f)
            o = sigmoid(w_o * input + U_o * h_t_1 + b_o)
            candidate = tanh(W * input + U * h_t_1 + b)
            c_t = i \odot candidate + f \odot c_t_1
            h_t = o \odot tanh(c_t)
        """
        i, o, f, u = F.sigmoid(i), F.sigmoid(o), F.sigmoid(f), F.tanh(u)

        c = torch.mul(i, u) + torch.mul(f, c_t_1)

        if self.use_o:
            h = torch.mul(o, F.tanh(c))
        else:
            h = F.tanh(c)
        return c, h

    def forward(self, inputs):
        c_t, h_t = None, None
        for i in range(inputs.size(0)):
            if i == 0:
                c_t_1 = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
                h_t_1 = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            else:
                c_t_1 = c_t
                h_t_1 = h_t
            c_t, h_t = self.node_forward(c_t_1, h_t_1, inputs[i])
        return c_t, h_t

class DualMLP(nn.Module):
    def __init__(self, rep_dim, hidden_dim, output_dim):
        super(DualMLP, self).__init__()
        self.rep_dim = rep_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer_1 = nn.Linear(2 * rep_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        output = self.layer_1(vec_dist)
        output = F.relu(output)
        output = self.layer_2(output)
        return output

class SentPairNetwork(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dims, type, sparsity, tune, use_o):
        super(SentPairNetwork, self).__init__()
        self.vocab_size = vocab_size
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.hidden_dims = hidden_dims
        self.type = type
        self.sparsity = sparsity
        self.tune = tune
        self.use_o = use_o

        # Embedding Layer
        self.emb = nn.Embedding(vocab_size, in_dim, sparse=sparsity)
        if not tune:
            self.emb.weight.requires_grad = False
        # Sequence Model
        if type == 'ulstm':
            self.model = ULSTM(in_dim, mem_dim, use_o)
            rep_dim = mem_dim
        elif type == 'plstm':
            self.model = PLSTM(in_dim, mem_dim, use_o)
            rep_dim = mem_dim
        elif type == 'lstm':
            self.model = LSTM(in_dim, mem_dim, use_o, peephole)
            rep_dim = mem_dim
        else:
            raise Exception('unsupported structure')
        self.mlp = DualMLP(rep_dim, hidden_dims[0], hidden_dims[1])

    def forward(self, linput, rinput):
        linput = self.emb(linput)
        rinput = self.emb(rinput)
        _, lrep = self.model(linput)
        _, rrep = self.model(rinput)
        output = self.mlp(lrep, rrep)
        return output
