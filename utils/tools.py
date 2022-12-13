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
    """Computes and stores the average and current value.

    Examples::
        # >>> # Initialize a meter to record loss
        # >>> losses = AverageMeter()
        # >>> # Update meter after every minibatch update
        # >>> losses.update(loss_value, batch_size)
    """

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


def accuracy(logits, target, topk=(1,), ignore_index=255):
    """ Suppose you have the ground truth prediction tensor y of shape b-h-w
        (dtype=torch.int64). Your model predicts per-pixel class logits of
        shape b-c-h-w, with c is the number of classes (including "background").
        These logits are the "raw" predictions before softmax function transforms
         them into class probabilities. Since we are only looking at the top k,
         it does not matter if the predictions are "raw" or "probabilities".
    """
    maxk = max(topk)

    # compute the top k predicted classes, per pixel
    _, tk = torch.topk(logits, maxk, dim=1)

    # you now have k predictions per pixel, and you want that one of them
    # will match the true labels target
    correct_pixels = torch.eq(target[:, None, ...], tk).any(dim=1).float()

    # take the mean of correct_pixels to get the overall average top-k accuracy
    valid = target != ignore_index
    top_k_acc = correct_pixels[valid].mean()

    return top_k_acc


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
