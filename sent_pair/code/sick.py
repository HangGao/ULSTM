from __future__ import division
from __future__ import print_function
import os
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import *
from vocab import Vocab
from dataset import read_data, UNK_WORD
from metrics import *
from utils import load_word_vectors, build_vocab, map_label_to_target
from config import parse_args
from tqdm import tqdm
import numpy

class KLCriterionPackage(object):
    def __init__(self, num_classes):
        self.criterion = nn.KLDivLoss()
        self.num_classes = num_classes

    def preprocess(self, x):
        return map_label_to_target(x, self.num_classes)

    def process(self, x):
        return F.log_softmax(x, dim=1)

    def postprocess(self, x):
        return torch.dot(torch.arange(1, self.num_classes + 1), torch.exp(x))

    def loss(self, output, target):
        return self.criterion(output, target)

    def cuda(self):
        self.criterion = self.criterion.cuda()

def create_logger():
    ############################################################################
    # global logger
    ############################################################################
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")

    ############################################################################
    # file logger
    ############################################################################
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ############################################################################
    # console logger
    ############################################################################
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def train_on_data(epoch, model, optimizer, criterion, dataset, args):
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    # random permutate the data set
    indices = torch.randperm(len(dataset))
    for idx in tqdm(range(len(dataset)),desc='Training epoch ' + str(epoch + 1) + ''):
        ltree, lsent, rtree, rsent, label = dataset[indices[idx]]

        linput, rinput = Var(lsent), Var(rsent)
        target = Var(criterion.preprocess(label))
        # put data on GPU if necessary
        if args.cuda:
            linput, rinput = linput.cuda(), rinput.cuda()
            target = target.cuda()

        output = model(linput, rinput)
        output = criterion.process(output)

        loss = criterion.loss(output, target)
        total_loss += loss.item()
        loss.backward()

        if idx % args.batchsize == 0 and idx > 0:
            optimizer.step()
            optimizer.zero_grad()

    if len(dataset) % args.batchsize != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataset)

def test_on_data(epoch, model, criterion, dataset, args):
    model.eval()

    total_loss = 0
    predictions = torch.zeros(len(dataset))
    for idx in tqdm(range(len(dataset)),desc='Testing epoch  ' + str(epoch) + ''):
        ltree, lsent, rtree, rsent, label = dataset[idx]

        linput, rinput = Var(lsent), Var(rsent)
        target = Var(criterion.preprocess(label))
        if args.cuda:
            linput, rinput = linput.cuda(), rinput.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model(linput, rinput)
            output = criterion.process(output)

            loss = criterion.loss(output, target)
            total_loss += loss.item()

            output = output.data.squeeze().cpu()
            predictions[idx] = criterion.postprocess(output)

    return total_loss / len(dataset), predictions

def run(epochs, model, optimizer, criterion, train_dataset, dev_dataset, logger, args):
    best = -float('inf')
    for epoch in range(epochs):
        train_loss             = train_on_data(epoch, model, optimizer, criterion, train_dataset, args)
        dev_loss, dev_pred     = test_on_data(epoch, model, criterion, dev_dataset, args)

        dev_pearson = pearson(dev_pred, dev_dataset.labels)
        dev_spearman = spearman(dev_pred, dev_dataset.labels)
        dev_mse = mse(dev_pred, dev_dataset.labels)
        logger.info('==> Epoch {}, Dev \tLoss: {}\tPearson: {}\tSpearman: {}\tMSE: {}'.format(epoch, dev_loss, dev_pearson, dev_spearman, dev_mse))

        if best < dev_pearson:
            best = dev_pearson
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optimizer,
                'pearson': dev_pearson,
                'spearman': dev_spearman,
                'mse': dev_mse,
                'args': args
            }
            logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))

# MAIN BLOCK
def main():
    global args
    args = parse_args()

    logger = create_logger()

    ############################################################################
    # argument validation
    ############################################################################
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()

    ############################################################################
    # set random seeds
    ############################################################################
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    ############################################################################
    # create all io directories
    ############################################################################
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    ############################################################################
    # load task vocab
    ############################################################################
    data_dir = 'data'
    task_dir = os.path.join(data_dir, args.data)
    vocab = Vocab(filename=os.path.join(task_dir, 'vocab-cased.txt'), data=[UNK_WORD])
    logger.debug('==> Task vocabulary size : %d ' % vocab.size())

    ############################################################################
    # load embeddings
    ############################################################################
    glove_vocab, glove_emb = load_word_vectors(os.path.join(data_dir, 'glove', 'glove.840B.300d'))
    logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
    emb = torch.Tensor(vocab.size(), glove_emb.size(1)).normal_(-0.05, 0.05)
    for word in vocab.labelToIdx.keys():
        if glove_vocab.getIndex(word):
            emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]

    ############################################################################
    # read data set
    ############################################################################
    train_dir = os.path.join(data_dir, args.data, 'train')
    dev_dir = os.path.join(data_dir, args.data, 'dev')
    test_dir = os.path.join(data_dir, args.data, 'test')

    train_set = read_data(args.data, train_dir, vocab)
    logger.debug('==> Size of train data   : %d ' % len(train_set))
    dev_set = read_data(args.data, dev_dir, vocab)
    logger.debug('==> Size of dev data     : %d ' % len(dev_set))
    if os.path.exists(test_dir):
        test_set = read_data(args.data, test_dir, vocab)
        logger.debug('==> Size of test data     : %d ' % len(test_set))
    else:
        test_set = None
        logger.debug('==> No test set is specified')


    ############################################################################
    # model, train, and evaluation
    ############################################################################
    network = SentPairNetwork(vocab.size(), args.input_dim, args.mem_dim, [args.hidden_dim, args.num_classes],
            args.model_type, args.sparse, args.tune_embs, args.use_o)
    criterion = KLCriterionPackage(args.num_classes)
    if args.cuda:
        network.cuda()
        criterion.cuda()
        emb = emb.cuda()
    # initiate model parameters
    network.emb.weight.data.copy_(emb)
    network.model.reset()

    params = list(network.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Model total parameters:', total_params)

    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, network.parameters()), lr=args.lr, weight_decay=args.wd)
    run(args.epochs, network, optimizer, criterion, train_set, dev_set, logger, args)

    ############################################################################
    # load the best model on dev set for testing
    ############################################################################
    if test_set is not None:
        checkpoint = torch.load('%s.pt' % os.path.join(args.save, args.expname))
        network.load_state_dict(checkpoint['model'])

        test_loss, test_pred = test_on_data(0, network, criterion, test_set, args)
        test_pearson = pearson(test_pred, test_set.labels)
        test_spearman = spearman(test_pred, test_set.labels)
        test_mse = mse(test_pred, test_set.labels)
        logger.info('==> Test \tPearson: {}\tSpearman: {}\tMSE: {}'.format(test_pearson, test_spearman, test_mse))

if __name__ == "__main__":
    main()
