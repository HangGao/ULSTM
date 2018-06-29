import os
import re
import torch
import numpy as np
from collections import Counter

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
