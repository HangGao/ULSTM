import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model
import sys
from utils import get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch FPLSTM/LSTM Sentence Pair Model')
parser.add_argument('--data', type=str, default='data/sick/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Define Corpus and Dictionary
###############################################################################
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        slef.dictionary.add_word('<PAD>')

        self.train = self.tokenize(os.path.join(path, 'SICK_train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'SICK_trial.txt'))
        self.test = self.tokenize(os.path.join(path, 'SICK_test_annotated.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            f.readline()

            max_sent_len = 0
            num_pairs = 0
            for line in f:
                entries = line.strip().split('\t')
                a, b, score = entries[1], entries[2], float(entries[3])
                num_pairs += 1

                # add left sentence words
                words = a.split()
                for word in words:
                    self.dictionary.add_word(word)
                max_sent_len = max(max_sent_len, len(words))

                # add right sentence words
                words = b.split()
                for word in words:
                    self.dictionary.add_word(word)
                max_sent_len = max(max_sent_len, len(words))

        # Tokenize file content
        with open(path, 'r') as f:
            f.readline()

            lsents = torch.LongTensor(num_pairs, max_sent_len).fill_(self.dictionary.word2idx('<PAD>'))
            rsents = torch.LongTensor(num_pairs, max_sent_len).fill_(self.dictionary.word2idx('<PAD>'))
            labels = torch.Tensor(num_pairs)

            for i in range(num_pairs):
                line = f.readline()
                entries = line.strip().split('\t')
                a, b, score = entries[1], entries[2], float(entries[3])

                # left sentence
                words = a.split()
                for j in range(len(words)):
                    lsents[i, j] = self.dictionary.word2idx[words[j]]

                # right sentence
                words = b.split()
                for j in range(len(words)):
                    rsents[i, j] = self.dictionary.word2idx[words[j]]

                labels[i] = float(score)

        return lsents, rsents, labels


###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

def batchify(data, bsz, args):
    lsents, rsents, labels = data

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = labels.size(0) // bsz

    # random sampling data to complement if not fit.
    if nbatch * bsz < labels.size(0):
        random_idx = np.random.permutation(labels.size(0))[:labels.size(0)-(nbatch*bsz)]
        lcomplement = torch.LongTensor(random_idx.shape[0])
        rcomplement = torch.LongTensor(random_idx.shape[0])
        labelcomplement = torch.Tensor(random_idx.shape[0])

        for i in range(random_idx.shape[0]):
            lcomplement[i] = lsents[random_idx[i]]
            rcomplement[i] = rsents[random_idx[i]]
            labelcomplement[i] = labels[random_idx[i]]

        lsents = torch.cat([lsents, lcomplement], 0)
        rsents = torch.cat([rsents, rcomplement], 0)
        labels = torch.cat([labels, labelcomplement], 0)

    lsents = lsents.view(bsz, -1).t().contiguous()  #(num_batch, batch_size)
    rsents = rsents.view(bsz, -1).t().contiguous()  #(num_batch, batch_size)
    labels = labels.view(bsz, -1).t().contiguous()  #(num_batch, batch_size)

    if args.cuda:
        lsents = lsents.cuda()
        rsents = rsents.cuda()
        labels = labels.cuda()

    return lsents, rsents, labels

eval_batch_size = 1
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################
num_classes = 5
class Network(nn.Module):

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, sim_dim=50, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0):
        super(Network, self).__init__()
        self.rnn = model.RNNModel(rnn_type, ntoken, ninp, nhid, nlayers, sim_dim, dropout, dropouth, dropouti, dropoute, wdrop, tie_weights=False)
        self.layer_1 = nn.Linear(2 * nhid, sim_dim)
        self.layer_2 = nn.Linear(sim_dim, num_classes)

    def forward(self, lsents, lmasks, lhids, rsents, rmasks, rhids):
        """
            lsents: (seq_len, batch_size)
            lmasks: (seq_len, batch_size)
            lhids: (batch_size, nhid)
            rsents: (seq_len, batch_size)
            rmasks: (seq_len, batch_size)
            rhids: (batch_size, nhid)
        """
        batch_idx = lsents.data.new(lsents.size(1)).copy_(torch.range(0, lsents.size(1)))

        loutput, lhidden, lrnn_hs, ldropped_rnn_hs = self.rnn(lsents, lhids, return_h=True)
        # last layer output (seq_len, batch_size, nhid)
        loutput = ldropped_rnn_hs[-1]
        loutput = torch.transpose(loutput, 0, 1)    # (batch_size, seq_len, nhid)
        # get the last output  (batch_size)
        last_idx = torch.sum(torch.gt(lsents, 0).type(type(lsents)), 0)
        loutput = loutput[last_idx, batch_idx]
        pass

criterion = nn.KLCriterionPackage(num_classes)
ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop

params = list(model.parameters()) + list(criterion.parameters())
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
    params = list(model.parameters()) + list(criterion.parameters())
###
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################

# def evaluate(data_source, batch_size=10):
#     # Turn on evaluation mode which disables dropout.
#     model.eval()
#     if args.model == 'QRNN': model.reset()
#     total_loss = 0
#     ntokens = len(corpus.dictionary)
#     hidden = model.init_hidden(batch_size)
#     for i in range(0, data_source.size(0) - 1, args.bptt):
#         data, targets = get_batch(data_source, i, args, evaluation=True)
#         output, hidden = model(data, hidden)
#         total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
#         hidden = repackage_hidden(hidden)
#     return total_loss[0] / len(data_source)


# def train():
#     # Turn on training mode which enables dropout.
#     if args.model == 'QRNN': model.reset()
#     total_loss = 0
#     start_time = time.time()
#     ntokens = len(corpus.dictionary)
#     hidden = model.init_hidden(args.batch_size)
#     batch, i = 0, 0
#     while i < train_data.size(0) - 1 - 1:
#         bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
#         # Prevent excessively small or negative sequence lengths
#         seq_len = max(5, int(np.random.normal(bptt, 5)))
#         # There's a very small chance that it could select a very long sequence length resulting in OOM
#         # seq_len = min(seq_len, args.bptt + 10)
#
#         lr2 = optimizer.param_groups[0]['lr']
#         optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
#         model.train()
#         data, targets = get_batch(train_data, i, args, seq_len=seq_len)
#
#         # Starting each batch, we detach the hidden state from how it was previously produced.
#         # If we didn't, the model would try backpropagating all the way to start of the dataset.
#         hidden = repackage_hidden(hidden)
#         optimizer.zero_grad()
#
#         output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
#         raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
#
#         loss = raw_loss
#         # Activiation Regularization
#         if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
#         # Temporal Activation Regularization (slowness)
#         if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
#         loss.backward()
#
#         # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
#         if args.clip: torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
#         optimizer.step()
#
#         total_loss += raw_loss.data
#         optimizer.param_groups[0]['lr'] = lr2
#         if batch % args.log_interval == 0 and batch > 0:
#             cur_loss = total_loss[0] / args.log_interval
#             elapsed = time.time() - start_time
#             print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
#                     'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
#                 epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
#                 elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
#             total_loss = 0
#             start_time = time.time()
#         ###
#         batch += 1
#         i += seq_len
#
# # Loop over epochs.
# lr = args.lr
# best_val_loss = []
# stored_loss = 100000000
#
# # At any point you can hit Ctrl + C to break out of training early.
# try:
#     optimizer = None
#     if args.optimizer == 'sgd':
#         optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
#     if args.optimizer == 'adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
#
#     for epoch in range(1, args.epochs+1):
#         epoch_start_time = time.time()
#         train()
#         if 't0' in optimizer.param_groups[0]:
#             tmp = {}
#             for prm in model.parameters():
#                 tmp[prm] = prm.data.clone()
#                 prm.data = optimizer.state[prm]['ax'].clone()
#
#             val_loss2 = evaluate(val_data)
#             print('-' * 89)
#             print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
#                 'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
#               epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
#             print('-' * 89)
#
#             if val_loss2 < stored_loss:
#                 model_save(args.save)
#                 print('Saving Averaged!')
#                 stored_loss = val_loss2
#
#             for prm in model.parameters():
#                 prm.data = tmp[prm].clone()
#
#         else:
#             val_loss = evaluate(val_data, eval_batch_size)
#             print('-' * 89)
#             print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
#                 'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
#               epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
#             print('-' * 89)
#
#             if val_loss < stored_loss:
#                 model_save(args.save)
#                 print('Saving model (new best validation)')
#                 stored_loss = val_loss
#
#             if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
#                 print('Switching to ASGD')
#                 optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
#
#             if epoch in args.when:
#                 print('Saving model before learning rate decreased')
#                 model_save('{}.e{}'.format(args.save, epoch))
#                 print('Dividing learning rate by 10')
#                 optimizer.param_groups[0]['lr'] /= 10.
#
#             best_val_loss.append(val_loss)
#
# except KeyboardInterrupt:
#     print('-' * 89)
#     print('Exiting from training early')
#
# # Load the best saved model.
# model_load(args.save)
#
# # Run on valid data.
# valid_loss = evaluate(val_data)
# print('=' * 89)
# print('| End of training | valid loss {:5.2f} | valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
#     valid_loss, math.exp(valid_loss), valid_loss / math.log(2)))
# print('=' * 89)
#
# # Run on test data.
# test_loss = evaluate(test_data, test_batch_size)
# print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
#     test_loss, math.exp(test_loss), test_loss / math.log(2)))
# print('=' * 89)
