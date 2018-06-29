import math
import torch

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
