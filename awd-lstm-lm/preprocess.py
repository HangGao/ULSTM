import os
import json
import random
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

def preprocess_yelp(dir, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):

    def write_record(fp, label, words):
        fp.write(label)
        fp.write('\t')
        fp.write(' '.join(words))
        fp.write('\n')

    dfile = os.path.join(dir, 'review.json')
    train_file = os.path.join(dir, 'train.txt')
    valid_file = os.path.join(dir, 'valid.txt')
    test_file = os.path.join(dir, 'test.txt')

    train_cnt, valid_cnt, test_cnt = 0, 0, 0
    train_fp, valid_fp, test_fp = open(train_file, 'w'), open(valid_file, 'w'), open(test_file, 'w')
    with open(dfile, 'r') as f:
        for line in f:
            record = json.loads(line)
            label = str(record['stars'])
            text = record['text']
            words = tokenizer.tokenize(text)

            die = random.random()
            # sample 10% of data
            if die > 0.1: continue
            die = random.random()

            if die < train_ratio:
                train_cnt += 1
                write_record(train_fp, label, words)
            elif die < train_ratio + dev_ratio:
                valid_cnt += 1
                write_record(valid_fp, label, words)
            else:
                test_cnt += 1
                write_record(test_fp, label, words)
            print('Train: %d, Valid:%d, Test:%d\033[0G' % (train_cnt, valid_cnt, test_cnt), end='\r')
    train_fp.close()
    valid_fp.close()
    test_fp.close()

yelp_path = os.path.join('data', 'yelp')
preprocess_yelp(yelp_path)
