from copy import deepcopy
import torch
from scipy.stats import spearmanr, pearsonr

def pearson(predictions, labels):
    a = predictions.numpy()
    b = labels.numpy()
    return pearsonr(a, b)[0]

def spearman(predictions, labels):
    a = predictions.numpy()
    b = labels.numpy()
    return spearmanr(a, b)[0]

def mse(predictions, labels):
    x = deepcopy(predictions)
    y = deepcopy(labels)
    return torch.mean((x - y) ** 2)

def acc(predictions, labels):
    return torch.sum(torch.eq(predictions, labels)) / predictions.size(0)


def precision(predictions, labels):
    tp = 0.
    fp = 0.
    fn = 0.
    tn = 0.
    for i in range(predictions.size(0)):
        if labels[i] == 1 and predictions[i] == 1:
            tp = tp + 1
        elif labels[i] == 1 and predictions[i] == 0:
            fn = fn + 1
        elif labels[i] == 0 and predictions[i] == 1:
            fp = fp + 1
        elif labels[i] == 0 and predictions[i] == 0:
            tn = tn + 1
    return tp / (tp + fp)

def recall(predictions, labels):
    tp = 0.
    fp = 0.
    fn = 0.
    tn = 0.
    for i in range(predictions.size(0)):
        if labels[i] == 1 and predictions[i] == 1:
            tp = tp + 1
        elif labels[i] == 1 and predictions[i] == 0:
            fn = fn + 1
        elif labels[i] == 0 and predictions[i] == 1:
            fp = fp + 1
        elif labels[i] == 0 and predictions[i] == 0:
            tn = tn + 1
    return tp / (tp + fn)

def f1(predictions, labels):
    p = precision(predictions, labels)
    r = recall(predictions, labels)
    return (2 * p * r) / (p + r)
