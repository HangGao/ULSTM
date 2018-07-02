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

class FPLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_o=True, peephole=False):
        super(FPLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.use_o = use_o
        self.peephole = peephole

        self.iozfux = nn.Linear(self.in_dim, 5 * self.mem_dim)
        self.iozfh = nn.Linear(self.mem_dim, 4 * self.mem_dim)
        self.um = nn.Linear(self.mem_dim, self.mem_dim)

        if peephole:
            self.ic = CMul(self.mem_dim)
            self.fc = CMul(self.mem_dim)
            self.oc = CMul(self.mem_dim)
            self.zc = CMul(self.mem_dim)

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
        if self.peephole:
            """
            Transitions:
                i = sigmoid(W_i * input + U_i * h_t_1 + p_i \odot c_t_1 + b_i)
                f = sigmoid(W_f * input + U_f * h_t_1 + p_f \odot c_t_1 + b_f)
                o = sigmoid(w_o * input + U_o * h_t_1 + p_o \odot c_t + b_o)
                z = sigmoid(W_z * input + U_z * h_t_1 + p_z \odot c_t_1 + b_z)
                candidate = tanh(W * input + M * (z \odot tanh(c_t_1)) + b)
                c_t = i \odot candidate + f \odot c_t_1
                h_t = o \odot tanh(c_t)
            """
            i, o, f = F.sigmoid(ix + ih + self.ic(c_t_1)), ox + oh, F.sigmoid(fx + fh + self.fc(c_t_1))

            z = zx + zh + self.zc(c_t_1)
            z = F.sigmoid(z)
        else:
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
            if self.peephole:
                o = F.sigmoid(o + self.oc(c))
            h = torch.mul(o, F.tanh(c))
        else:
            h = F.tanh(c)
        return c, h

    def _forward(self, inputs, back=False):
        if back:
            c_t, h_t = None, None
            for i in range(inputs.size(0)-1, -1, -1):
                if i == inputs.size(0)-1:
                    c_t_1 = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
                    h_t_1 = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
                else:
                    c_t_1 = c_t
                    h_t_1 = h_t
                c_t, h_t = self.node_forward(c_t_1, h_t_1, inputs[i])
            return c_t, h_t
        else:
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

    def forward(self, inputs, back=False):
        c, h = self._forward(inputs, back=back)
        return h

class BiFPLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_o=True, peephole=False):
        super(BiFPLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.use_o = use_o
        self.peephole = peephole

        self.fplstm = FPLSTM(in_dim, mem_dim, use_o, peephole)

    def reset(self):
        self.fplstm.reset()

    def forward(self, inputs):
        frep = self.fplstm(inputs)
        brep = self.fplstm(inputs, back=True)
        rep = torch.cat([frep, brep], dim=1)
        return rep

class LSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_o=True, peephole=False):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.use_o = use_o
        self.peephole = peephole

        self.iofux = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.iofuh = nn.Linear(self.mem_dim, 4 * self.mem_dim)

        if self.peephole:
            self.ic = CMul(self.mem_dim)
            self.fc = CMul(self.mem_dim)
            self.oc = CMul(self.mem_dim)

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
        if self.peephole:
            """
            Transitions:
                i = sigmoid(W_i * input + U_i * h_t_1 + p_i \odot c_t_1 + b_i)
                f = sigmoid(W_f * input + U_f * h_t_1 + p_f \odot c_t_1 + b_f)
                o = sigmoid(w_o * input + U_o * h_t_1 + p_o \odot c + b_o)
                candidate = tanh(W * input + U * h_t_1 + b)
                c_t = i \odot candidate + f \odot c_t_1
                h_t = o \odot tanh(c_t)
            """
            i, f, u = F.sigmoid(i + self.ic(c_t_1)), F.sigmoid(f + self.fc(c_t_1)), F.tanh(u)
        else:
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
            if self.peephole:
                o = F.sigmoid(o + self.oc(c))
            h = torch.mul(o, F.tanh(c))
        else:
            h = F.tanh(c)
        return c, h

    def _forward(self, inputs, back=False):
        if back:
            c_t, h_t = None, None
            for i in range(inputs.size(0)-1, -1, -1):
                if i == inputs.size(0)-1:
                    c_t_1 = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
                    h_t_1 = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
                else:
                    c_t_1 = c_t
                    h_t_1 = h_t
                c_t, h_t = self.node_forward(c_t_1, h_t_1, inputs[i])
            return c_t, h_t
        else:
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

    def forward(self, inputs, back=False):
        c, h = self._forward(inputs, back=back)
        return h

class BiLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_o=True, peephole=False):
        super(BiLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.use_o = use_o
        self.peephole = peephole

        self.lstm = LSTM(in_dim, mem_dim, use_o, peephole)

    def reset(self):
        self.lstm.reset()

    def forward(self, inputs):
        frep = self.lstm(inputs)
        brep = self.lstm(inputs, back=True)
        rep = torch.cat([frep, brep], dim=1)
        return rep

class FPChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_o=True, peephole=False):
        super(FPChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.use_o = use_o
        self.peephole = peephole

        self.ifozux = nn.Linear(self.in_dim, 5 * self.mem_dim)
        self.ios = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.fzh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.um = nn.Linear(self.mem_dim, self.mem_dim)

        if peephole:
            self.ic = CMul(self.mem_dim)
            self.fc = CMul(self.mem_dim)
            self.oc = CMul(self.mem_dim)
            self.zc = CMul(self.mem_dim)

    def reset(self):
        self.ifozux.bias.data[self.mem_dim : 2*self.mem_dim].fill_(0.5)
        self.fzh.bias.data[: self.mem_dim].fill_(0.5)

    def node_forward(self, child_c, child_h, input):
        '''
        child_c   -->   (k, mem_dim)
        child_h   -->   (k, mem_dim)
        input   -->   (1, in_dim)
        '''
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True) # (1, mem_dim)

        ifozu = torch.unsqueeze(self.ifozux(input), 0)
        ix, fx, ox, zx, ux = torch.split(ifozu, ifozu.size(1) // 5, dim=1)

        io = self.ios(child_h_sum)
        i_s, o_s = torch.split(io, io.size(1) // 2, dim=1)

        fz = self.fzh(child_h)
        fh, zh = torch.split(fz, fz.size(1) // 2, dim=1)

        i = ix + i_s
        f = fh + fx.repeat(child_h.size(0), 1)
        o = ox + o_s
        z = zx.repeat(child_h.size(0), 1) + zh

        if self.peephole:
            """
            Transitions:
                child_sum_h = sum (child_h, dim=0)
                i = sigmoid(W_i * input + U_i * child_sum_h + p_i \odot child_sum_c + b_i)
                f_j = sigmoid(W_f * input + U_f * child_h_j + p_f \odot child_c_j + b_f)
                o = sigmoid(w_o * input + U_o * child_sum_h + p_o \odot c + b_o)
                z_j = sigmoid(W_z * input + U_z * child_h_j + p_z \odot child_c_j + b_z)
                candidate = tanh(W * input + M * (sum(z_j \odot tanh(child_c_j))))
                c = i \odot candidate + sum(f_j \odot child_c_j)
                h = o \odot tanh(c)
            """
            i = i + self.ic(torch.sum(child_c, dim=0, keepdim=True))
            i = F.sigmoid(i)
            f = f + self.fc(child_c)
            f = F.sigmoid(f)
            z = z + self.zc(child_c)
            z = F.sigmoid(z)
        else:
            """
            Transitions:
                child_sum_h = sum (child_h, dim=0)
                i = sigmoid(W_i * input + U_i * child_sum_h + b_i)
                f_j = sigmoid(W_f * input + U_f * child_h_j + b_f)
                o = sigmoid(w_o * input + U_o * child_sum_h + b_o)
                z_j = sigmoid(W_z * input + U_z * child_h_j + M_z * tanh(child_c_j) + b_z)
                candidate = tanh(W * input + M * (sum(z_j \odot tanh(child_c_j))))
                c = i \odot candidate + sum(f_j \odot child_c_j)
                h = o \odot tanh(c)
            """
            i = F.sigmoid(i)
            f = F.sigmoid(f)
            o = F.sigmoid(o)
            z = F.sigmoid(z)

        u = ux + self.um(torch.sum(torch.mul(z, F.tanh(child_c)), dim=0, keepdim=True))
        u = F.tanh(u)

        c = torch.mul(i, u) + torch.sum(torch.mul(f, child_c), dim=0, keepdim=True)

        if self.use_o:
            if self.peephole:
                o = F.sigmoid(o + self.oc(c))
            h = torch.mul(o, F.tanh(c))
        else:
            h = F.tanh(c)
        return c, h

    def _forward(self, tree, inputs):
        _ = [self._forward(tree.children[idx], inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:
            child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(child_c, child_h, inputs[tree.idx])
        return tree.state

    def forward(self, tree, inputs):
        cell, hidden = self._forward(tree, inputs)
        return hidden

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_o=True, peephole=False):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.use_o = use_o
        self.peephole = peephole

        self.ifoux = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

        if self.peephole:
            self.ic = CMul(self.mem_dim)
            self.fc = CMul(self.mem_dim)
            self.oc = CMul(self.mem_dim)

    def reset(self):
        self.ifoux.bias.data[self.mem_dim : 2*self.mem_dim].fill_(0.5)
        self.fh.bias.data.fill_(0.5)

    def node_forward(self, child_c, child_h, input):
        '''
        child_c   -->   (k, mem_dim)
        child_h   -->   (k, mem_dim)
        input   -->   (1, in_dim)
        '''
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True) # (1, mem_dim)

        ifou = torch.unsqueeze(self.ifoux(input), 0)
        ix, fx, ox, ux = torch.split(ifou, ifou.size(1) // 4, dim=1)

        iou = self.ious(child_h_sum)
        i_s, o_s, u_s = torch.split(iou, iou.size(1) // 3, dim=1)

        i = ix + i_s
        f = self.fh(child_h) + fx.repeat(child_h.size(0), 1)
        o = ox + o_s
        u = ux + u_s
        u = F.tanh(u)

        if self.peephole:
            """
            Transitions:
                child_sum_h = sum (child_h, dim=0)
                i = sigmoid(W_i * input + U_i * child_sum_h + p_i \odot child_sum_c + b_i)
                f_j = sigmoid(W_f * input + U_f * child_h_j + p_f \odot child_c_j + b_f)
                o = sigmoid(w_o * input + U_o * child_sum_h + p_o \odot c + b_o)
                candidate = tanh(W * input + U * child_sum_h + b)
                c = i \odot candidate + sum(f_j \odot child_c_j)
                h = o \odot tanh(c)
            """
            i = i + self.ic(torch.sum(child_c, dim=0, keepdim=True))
            i = F.sigmoid(i)
            f = f + self.fc(child_c)
            f = F.sigmoid(f)
        else:
            """
            Transitions:
                child_sum_h = sum (child_h, dim=0)
                i = sigmoid(W_i * input + U_i * child_sum_h + b_i)
                f_j = sigmoid(W_f * input + U_f * child_h_j + b_f)
                o = sigmoid(w_o * input + U_o * child_sum_h + b_o)
                candidate = tanh(W * input + U * child_sum_h + b)
                c = i \odot candidate + sum(f_j \odot child_c_j)
                h = o \odot tanh(c)
            """
            i = F.sigmoid(i)
            f = F.sigmoid(f)
            o = F.sigmoid(o)

        c = torch.mul(i, u) + torch.sum(torch.mul(f, child_c), dim=0, keepdim=True)

        if self.use_o:
            if self.peephole:
                o = F.sigmoid(o + self.oc(c))
            h = torch.mul(o, F.tanh(c))
        else:
            h = F.tanh(c)
        return c, h

    def _forward(self, tree, inputs):
        _ = [self._forward(tree.children[idx], inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:
            child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(child_c, child_h, inputs[tree.idx])
        return tree.state

    def forward(self, tree, inputs):
        cell, hidden = self._forward(tree, inputs)
        return hidden

class FPChainTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_o, peephole):
        super(FPChainTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.use_o = use_o
        self.peephole = peephole

        self.iofzux = nn.Linear(self.in_dim, 5 * self.mem_dim)
        self.iofzh = nn.Linear(self.mem_dim, 4 * self.mem_dim)

        self.um = nn.Linear(self.mem_dim, self.mem_dim)

        if self.peephole:
            self.ic = CMul(self.mem_dim)
            self.fc = CMul(self.mem_dim)
            self.oc = CMul(self.mem_dim)
            self.zc = CMul(self.mem_dim)

    def reset(self):
        self.iofzux.bias.data[2 * self.mem_dim : 3 * self.mem_dim].fill_(0.5)
        self.iofzh.bias.data[2 * self.mem_dim : 3 * self.mem_dim].fill_(0.5)

    def node_forward(self, input, parent_c, parent_h):
        '''
        input   -->   (1, in_dim)
        parent_c   -->   (1, mem_dim)
        parent_h   -->   (1, mem_dim)
        '''
        iofzu = torch.unsqueeze(self.iofzux(input), 0)
        ix, ox, fx, zx, ux = torch.split(iofzu, iofzu.size(1) // 5, dim=1)

        iofz = self.iofzh(parent_h)
        ih, oh, fh, zh = torch.split(iofz, iofz.size(1) // 4, dim=1)

        i = ix + ih
        f = fx + fh
        o = ox + oh
        z = zx + zh

        if self.peephole:
            """
            Transitions:
                i = sigmoid(W_i * input + U_i * parent_h + p_i \odot parent_c + b_i)
                f = sigmoid(W_f * input + U_f * parent_h + p_f \odot parent_c + b_f)
                o = sigmoid(W_o * input + U_o * parent_h + p_o \odot c + b_o)
                z = sigmoid(W_z * input + U_z * parent_h + p_z \odot parent_c + b_z)
                candidate = tanh(W * input + M * (z \odot tanh(parent_c)) + b)
                c = i \odot candidate + f \odot parent_c
                h = o \odot tanh(c)
            """
            i = i + self.ic(parent_c)
            i = F.sigmoid(i)
            f = f + self.fc(parent_c)
            f = F.sigmoid(f)
            z = z + self.zc(parent_c)
            z = F.sigmoid(z)
        else:
            """
            Transitions:
                i = sigmoid(W_i * input + U_i * parent_h + b_i)
                f = sigmoid(W_f * input + U_f * parent_h + b_f)
                o = sigmoid(W_o * input + U_o * parent_h + b_o)
                z = sigmoid(W_z * input + U_z * parent_h + M_z * tanh(parent_c) + b_z)
                candidate = tanh(W * input + M * (z \odot tanh(parent_c)) + b)
                c = i \odot candidate + f \odot parent_c
                h = o \odot tanh(c)
            """
            i = F.sigmoid(i)
            f = F.sigmoid(f)
            o = F.sigmoid(o)
            z = F.sigmoid(z)

        u = ux + self.um(torch.mul(z, F.tanh(parent_c)))
        u = F.tanh(u)

        c = torch.mul(i, u) + torch.mul(f, parent_c)

        if self.use_o:
            if self.peephole:
                o = F.sigmoid(o + self.oc(c))
            h = torch.mul(o, F.tanh(c))
        else:
            h = F.tanh(c)
        return c, h

    def _forward(self, tree, inputs):
        if tree.parent is None:
            parent_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            parent_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        else:
            parent_c, parent_h = tree.parent.cstate

        tree.cstate = self.node_forward(inputs[tree.idx], parent_c, parent_h)

        child_states = [self._forward(tree.children[idx], inputs) for idx in range(tree.num_children)]
        if tree.num_children > 0:
            return [tree.cstate] + reduce(lambda x, y: x + y, child_states)
        else:
            return [tree.cstate]

    def forward(self, tree, inputs):
        states = self._forward(tree, inputs)
        state_h = [state[1] for state in states]
        mat = torch.cat(state_h, dim=0)
        state = torch.max(mat, dim=0, keepdim=True)[0]
        return state

class ChainTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_o, peephole):
        super(ChainTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.use_o = use_o
        self.peephole = peephole

        self.iofux = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.iofuh = nn.Linear(self.mem_dim, 4 * self.mem_dim)

        if self.peephole:
            self.ic = CMul(self.mem_dim)
            self.fc = CMul(self.mem_dim)
            self.oc = CMul(self.mem_dim)

    def reset(self):
        self.iofux.bias.data[2 * self.mem_dim : 3 * self.mem_dim].fill_(0.5)
        self.iofuh.bias.data[2 * self.mem_dim : 3 * self.mem_dim].fill_(0.5)

    def node_forward(self, input, parent_c, parent_h):
        '''
        input   -->   (1, in_dim)
        parent_c   -->   (1, mem_dim)
        parent_h   -->   (1, mem_dim)
        '''
        iofu = self.iofux(input) + self.iofuh(parent_h)
        i, o, f, u = torch.split(iofu, iofu.size(1) // 4, dim=1)

        if self.peephole:
            """
            Transitions:
                i = sigmoid(W_i * input + U_i * parent_h + p_i \odot parent_c + b_i)
                f = sigmoid(W_f * input + U_f * parent_h + p_f \odot parent_c + b_f)
                o = sigmoid(W_o * input + U_o * parent_h + p_o \odot c + b_o)
                candidate = tanh(W * input + U * parent_h + b)
                c = i \odot candidate + f \odot parent_c
                h = o \odot tanh(c)
            """
            i, f, u = F.sigmoid(i + self.ic(parent_c)), F.sigmoid(f + self.fc(parent_c)), F.tanh(u)
        else:
            """
            Transitions:
                i = sigmoid(W_i * input + U_i * parent_h + b_i)
                f = sigmoid(W_f * input + U_f * parent_h + b_f)
                o = sigmoid(W_o * input + U_o * parent_h + b_o)
                candidate = tanh(W * input + U * parent_h + b)
                c = i \odot candidate + f \odot parent_c
                h = o \odot tanh(c)
            """
            i, o, f, u = F.sigmoid(i), F.sigmoid(o), F.sigmoid(f), F.tanh(u)

        c = torch.mul(i, u) + torch.mul(f, parent_c)

        if self.use_o:
            if self.peephole:
                o = F.sigmoid(o + self.oc(c))
            h = torch.mul(o, F.tanh(c))
        else:
            h = F.tanh(c)
        return c, h

    def _forward(self, tree, inputs):
        if tree.parent is None:
            parent_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            parent_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        else:
            parent_c, parent_h = tree.parent.cstate

        tree.cstate = self.node_forward(inputs[tree.idx], parent_c, parent_h)

        child_states = [self._forward(tree.children[idx], inputs) for idx in range(tree.num_children)]
        if tree.num_children > 0:
            return [tree.cstate] + reduce(lambda x, y: x + y, child_states)
        else:
            return [tree.cstate]

    def forward(self, tree, inputs):
        states = self._forward(tree, inputs)
        state_h = [state[1] for state in states]
        mat = torch.cat(state_h, dim=0)
        state = torch.max(mat, dim=0, keepdim=True)[0]
        return state

class BiFPTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_o, peephole):
        super(BiFPTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.use_o = use_o
        self.peephole = peephole

        self.fmodel = FPChildSumTreeLSTM(in_dim, mem_dim, use_o, peephole)
        self.bmodel = FPChainTreeLSTM(in_dim, mem_dim, use_o, peephole)

    def reset(self):
        self.fmodel.reset()
        self.bmodel.reset()

    def forward(self, tree, inputs):
        frep = self.fmodel(tree, inputs)
        brep = self.bmodel(tree, inputs)

        rep = torch.cat([frep, brep], dim=1)
        return rep

class BiTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_o, peephole):
        super(BiTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.use_o = use_o
        self.peephole = peephole

        self.fmodel = ChildSumTreeLSTM(in_dim, mem_dim, use_o, peephole)
        self.bmodel = ChainTreeLSTM(in_dim, mem_dim, use_o, peephole)

    def reset(self):
        self.fmodel.reset()
        self.bmodel.reset()

    def forward(self, tree, inputs):
        frep = self.fmodel(tree, inputs)
        brep = self.bmodel(tree, inputs)

        rep = torch.cat([frep, brep], dim=1)
        return rep

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
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dims, type, sparsity, tune, use_o, peephole):
        super(SentPairNetwork, self).__init__()
        self.vocab_size = vocab_size
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.hidden_dims = hidden_dims
        self.type = type
        self.sparsity = sparsity
        self.tune = tune
        self.use_o = use_o
        self.peephole = peephole

        # Embedding Layer
        self.emb = nn.Embedding(vocab_size, in_dim, sparse=sparsity)
        if not tune:
            self.emb.weight.requires_grad = False
        # Sequence Model
        if type == 'fplstm':
            self.model = FPLSTM(in_dim, mem_dim, use_o, peephole)
            rep_dim = mem_dim
        elif type == 'bifplstm':
            self.model = BiFPLSTM(in_dim, mem_dim, use_o, peephole)
            rep_dim = 2 * mem_dim
        elif type == 'lstm':
            self.model = LSTM(in_dim, mem_dim, use_o, peephole)
            rep_dim = mem_dim
        elif type == 'bilstm':
            self.model = BiLSTM(in_dim, mem_dim, use_o, peephole)
            rep_dim = 2 * mem_dim
        elif type == 'fpchildsumtreelstm':
            self.model = FPChildSumTreeLSTM(in_dim, mem_dim, use_o, peephole)
            rep_dim = mem_dim
        elif type == 'childsumtreelstm':
            self.model = ChildSumTreeLSTM(in_dim, mem_dim, use_o, peephole)
            rep_dim = mem_dim
        elif type == 'fpchaintreelstm':
            self.model = FPChainTreeLSTM(in_dim, mem_dim, use_o, peephole)
            rep_dim = mem_dim
        elif type == 'chaintreelstm':
            self.model = ChainTreeLSTM(in_dim, mem_dim, use_o, peephole)
            rep_dim = mem_dim
        elif type == 'bitreelstm':
            self.model = BiTreeLSTM(in_dim, mem_dim, use_o, peephole)
            rep_dim = 2 * mem_dim
        elif type == 'bifptreelstm':
            self.model = BiFPTreeLSTM(in_dim, mem_dim, use_o, peephole)
            rep_dim = 2 * mem_dim
        else:
            raise Exception('unsupported structure')
        self.mlp = DualMLP(rep_dim, hidden_dims[0], hidden_dims[1])

    def forward(self, ltree, linput, rtree, rinput):
        linput = self.emb(linput)
        rinput = self.emb(rinput)
        if self.type in ['fplstm', 'lstm', 'bilstm', 'bifplstm']:
            lrep = self.model(linput)
            rrep = self.model(rinput)
        elif self.type in ['fpchildsumtreelstm', 'childsumtreelstm', 'fpchaintreelstm',
            'chaintreelstm', 'bitreelstm', 'bifptreelstm']:
            lrep = self.model(ltree, linput)
            rrep = self.model(rtree, rinput)
        else:
            raise Exception('unsupported structure')
        output = self.mlp(lrep, rrep)
        return output

# module for combining all of them together
class CrossEntropyTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, hidden_dim, num_classes, model_type, use_o):
        super(CrossEntropyTreeLSTM, self).__init__()
        self.model_type = model_type
        if model_type == 'fgbttreelstm':
            self.model = FGBTTreeLSTM(in_dim, mem_dim, use_o)
            self.similarity = CrossEntroyClassifier(mem_dim, hidden_dim, num_classes)
        elif model_type == 'fgtbtreelstm':
            self.model = FGTBTreeLSTM(in_dim, mem_dim, use_o)
            self.similarity = CrossEntroyClassifier(mem_dim, hidden_dim, num_classes)
        elif model_type == 'bifgtreelstm':
            self.model = BiFGTreeLSTM(in_dim, mem_dim, use_o)
            self.similarity = CrossEntroyClassifier(mem_dim, hidden_dim, num_classes, True)
        elif model_type == 'fglstm':
            self.model = FGLSTM(in_dim, mem_dim, use_o)
            self.similarity = CrossEntroyClassifier(mem_dim, hidden_dim, num_classes)
        elif model_type == 'bifglstm':
            self.model = FGLSTM(in_dim, mem_dim, use_o)
            self.similarity = CrossEntroyClassifier(mem_dim, hidden_dim, num_classes, True)
        else:
            raise Exception('unsupported structure')

    def forward(self, ltree, linputs, rtree, rinputs):
        if self.model_type == 'fgbttreelstm':
            lstate, lhidden = self.model(ltree, linputs)
            rstate, rhidden = self.model(rtree, rinputs)
            output = self.similarity(lhidden, rhidden)
        elif self.model_type == 'fgtbtreelstm' or self.model_type == 'bifgtreelstm':
            lhidden = self.model(ltree, linputs)
            rhidden = self.model(rtree, rinputs)
            output = self.similarity(lhidden, rhidden)
        elif self.model_type == 'fglstm':
            lstate, lhidden = self.model(linputs)
            rstate, rhidden = self.model(rinputs)
            output = self.similarity(lhidden, rhidden)
        elif self.model_type == 'bifglstm':
            lfstate, lfhidden = self.model(linputs)
            lbstate, lbhidden = self.model(linputs, True)
            lhidden = torch.cat((lfhidden, lbhidden), 1)
            rfstate, rfhidden = self.model(rinputs)
            rbstate, rbhidden = self.model(rinputs, True)
            rhidden = torch.cat((rfhidden, rbhidden), 1)
            output = self.similarity(lhidden, rhidden)
        return output
