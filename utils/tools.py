import numpy as np
import argparse
import torch
import os


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
                                topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.

    Examples::
        # >>> from torchreid import metrics
        # >>> metrics.accuracy(output, target)
    """

    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def eval_image(predict, label, num_classes):
    index = np.where((label >= 0) & (label < num_classes))
    predict = predict[index]
    label = label[index]

    TP = np.zeros((num_classes, 1))
    FP = np.zeros((num_classes, 1))
    TN = np.zeros((num_classes, 1))
    FN = np.zeros((num_classes, 1))

    for i in range(0, num_classes):
        TP[i] = np.sum(label[np.where(predict == i)] == i)
        FP[i] = np.sum(label[np.where(predict == i)] != i)
        TN[i] = np.sum(label[np.where(predict != i)] != i)
        FN[i] = np.sum(label[np.where(predict != i)] == i)

    return TP, FP, TN, FN, len(label)


def get_size_dataset():
    # folder path
    dir_path = os.path.join(os.getcwd(), 'data/img')
    count = 0

    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    return count
