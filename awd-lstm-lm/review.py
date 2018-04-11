import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter

import data
import model
import sys
from utils import batchify, get_batch, repackage_hidden
import os

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/sts/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, FPLSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=150,
                    help='number of hidden units per layer')
parser.add_argument('--num_classes', type=int, default=5,
                    help='number of classed')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=25, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.4,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=141,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
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
parser.add_argument('--do_not_pretrain', action='store_false',
                    help='do not use pretrain embeddings')
args = parser.parse_args()
args.tied = False

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
import re

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
        self.PAD_TOKEN = '<PAD>'
        self.dictionary.add_word(self.PAD_TOKEN)

        self.train, self.valid, self.test = self.read_data(path)

    def read_data(self, path):
        phrase2id = dict()
        with open(os.path.join(path, 'dictionary.txt'), 'r') as f:
            for line in f:
                line = re.sub(r'[^\x00-\x7F]',' ', line)
                phrase, id = line.strip().split('|')
                phrase2id[phrase] = id

        id2sentiment = dict()
        with open(os.path.join(path, 'sentiment_labels.txt'), 'r') as f:
            f.readline()
            for line in f:
                line = re.sub(r'[^\x00-\x7F]',' ', line)
                id, sentiment = line.strip().split('|')
                id2sentiment[id] = float(sentiment)

        sidx2text = dict()
        with open(os.path.join(path, 'datasetSentences.txt'), 'r') as f:
            f.readline()
            for line in f:
                line = re.sub(r'[^\x00-\x7F]',' ', line)
                idx, sent = line.strip().split('\t')
                sidx2text[idx] = sent

        train = []
        valid = []
        test = []
        max_sent_len = 0
        with open(os.path.join(path, 'datasetSplit.txt'), 'r') as f:
            f.readline()
            for line in f:
                line = re.sub(r'[^\x00-\x7F]',' ', line)
                idx, split = line.strip().split(',')
                text = sidx2text[idx]
                if text not in phrase2id: continue
                sentiment = id2sentiment[phrase2id[text]]
                if sentiment <= 0.2:
                    sentiment = 0
                elif sentiment <= 0.4:
                    sentiment = 1
                elif sentiment <= 0.6:
                    sentiment = 2
                elif sentiment <= 0.8:
                    sentiment = 3
                else:
                    sentiment = 4

                words = text.split()
                for word in words:
                    self.dictionary.add_word(word)
                max_sent_len = max(max_sent_len, len(words))

                if int(split) == 1:
                    train.append((sentiment, text))
                elif int(split) == 2:
                    test.append((sentiment, text))
                elif int(split) == 3:
                    valid.append((sentiment, text))
                else:
                    raise Exception('wrong data file')

        np.random.shuffle(train)

        def to_torch_tensor(dataset):
            targets = torch.LongTensor(len(dataset)).fill_(0)
            texts = torch.LongTensor(len(dataset), max_sent_len).fill_(0)

            for i in range(len(dataset)):
                sentiment, text = dataset[i]
                targets[i] = sentiment
                words = text.split()
                for j in range(len(words)):
                    texts[i, j] = self.dictionary.word2idx[words[j]]
            return texts, targets

        train = to_torch_tensor(train)
        valid = to_torch_tensor(valid)
        test = to_torch_tensor(test)

        return train, valid, test

def batchify(dataset, batch_size, args):
    data, target = dataset

    perm = torch.randperm(target.size(0))
    data = data[perm]
    target = target[perm]

    if args.cuda:
        data = data.cuda()
        target = target.cuda()

    n_batches = int(math.ceil(target.size(0) / batch_size))
    return (data, target), n_batches

def get_batch(dataset, batch, batch_size):
    data, target = dataset
    sidx = batch * batch_size
    eidx = min(target.size(0), (batch + 1) * batch_size)

    return data[sidx:eidx].t(), target[sidx:eidx]

###############################################################################
# Read Corpus
###############################################################################
def model_save(fn):
    global model, criterion, optimizer
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib
fn = 'corpus/corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus, embs = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = Corpus(args.data)

    print('Producing embeddings...')
    if not args.do_not_pretrain:
        embs = torch.Tensor(len(corpus.dictionary), args.emsize).uniform_(-0.05, 0.05)
    else:
        assert args.emsize == 300
        embs = torch.Tensor(len(corpus.dictionary), 300).uniform_(-0.05, 0.05)
        found = 0
        with open('data/glove/glove.840B.300d.txt', 'r') as f:
            for line in f:
                e = line.split()
                if len(e) > 301:
                    continue
                if e[0] in corpus.dictionary.word2idx:
                    arr = [float(x) for x in e[1:]]
                    embs[corpus.dictionary.word2idx[e[0]]] = torch.from_numpy(np.array(arr, 'float32'))
                    found += 1
                    print("%d out of %d found in glove vectors" % (found, len(corpus.dictionary)), end='\r')
    embs[0] = 0.
    torch.save((corpus, embs), fn)

###############################################################################
# Build the model
###############################################################################
class Network(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, num_classes, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0):
        super(Network, self).__init__()
        self.rnn = model.RNNModel(rnn_type, ntoken, ninp, nhid, nlayers, dropout, dropouth, dropouti, dropoute, wdrop, tie_weights=False, use_decoder=False)
        self.linear= nn.Linear(nhid, num_classes)

    def forward(self, sents, hids):
        """
            sents: (max_sent_len, batch_size)
        """
        output, hidden, rnn_hs, dropped_rnn_hs = self.rnn(sents, hids, return_h=True)
        hids = dropped_rnn_hs[-1]

        seq_len = hids.size(0)
        batch_len = hids.size(1)

        hids = hids.view(-1, hids.size(2))
        batch_idx = Variable(torch.arange(batch_len).type(type(sents.data)))
        seq_idx = torch.sum(torch.gt(sents, 0).type(type(sents.data)), 0) - 1
        idx = seq_idx * batch_len + batch_idx
        hids = torch.index_select(hids, 0, idx)

        output = F.log_softmax(self.linear(hids), 1)

        return output, rnn_hs, dropped_rnn_hs

criterion = nn.NLLLoss()
ntokens = len(corpus.dictionary)
model = Network(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.num_classes, args.dropout, \
                args.dropouth, args.dropouti, args.dropoute, args.wdrop)
model.rnn.encoder.weight.data.copy_(embs)
# model.rnn.encoder.weight.requires_grad = False

params = list(model.parameters()) + list(criterion.parameters())
params = filter(lambda p: p.requires_grad, params)
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
    params = list(model.parameters()) + list(criterion.parameters())
    params = filter(lambda p: p.requires_grad, params)
###
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

val_data, val_n_batches = batchify(corpus.valid, args.batch_size, args)
test_data, test_n_batches = batchify(corpus.test, args.batch_size, args)

##############################################################################
# Training code
##############################################################################
def evaluate(data_source, nbatch, batch_size=50):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)

    correct, total = 0., 0.
    for batch in range(nbatch):
        tbatch, lbatch = get_batch(data_source, batch, args.batch_size)
        lbatch, tbatch = Variable(lbatch), Variable(tbatch)

        # output (batch_size, num_classes)
        hidden = model.rnn.init_hidden(lbatch.size(0))
        output, rnn_hs, dropped_rnn_hs = model(tbatch, hidden)
        raw_loss = criterion(output, lbatch)
        total_loss += raw_loss.data

        output = torch.exp(output).data.cpu().numpy()
        prediction = np.argmax(output, 1)
        crt = np.asarray(np.equal(prediction, lbatch.data.cpu().numpy()), 'int32')
        correct += np.sum(crt)
        total += prediction.shape[0]

    return total_loss[0]/nbatch, correct / total


def train():
    # set global variables
    global optimizer, epoch

    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    train_data, nbatch = batchify(corpus.train, args.batch_size, args)
    for batch in range(nbatch):
        model.train()

        tbatch, lbatch = get_batch(train_data, batch, args.batch_size)
        lbatch, tbatch = Variable(lbatch), Variable(tbatch)
        hidden = model.rnn.init_hidden(lbatch.size(0))

        optimizer.zero_grad()

        # output (batch_size, num_classes)
        output, rnn_hs, dropped_rnn_hs = model(tbatch, hidden)
        raw_loss = criterion(output, lbatch)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm(filter(lambda p:p.requires_grad, model.parameters()), args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | loss {:5.2f} '.format(
                epoch, batch, nbatch, optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 10000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wdecay)

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()

        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2, val_acc2 = evaluate(val_data, val_n_batches)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid acc {:5.2f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss2, val_acc2))
            print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss, val_acc = evaluate(val_data, val_n_batches)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | val acc {:5.2f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, val_acc))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on valid data.
val_loss, val_acc = evaluate(val_data, val_n_batches)
print('=' * 89)
print('| End of training | valid loss {:5.2f} | valid acc {:5.2f}'.format(
    val_loss, val_acc))
print('=' * 89)

# Run on test data.
test_loss, test_acc = evaluate(test_data, test_n_batches)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test acc {:5.2f}'.format(
    test_loss, test_acc))
print('=' * 89)
