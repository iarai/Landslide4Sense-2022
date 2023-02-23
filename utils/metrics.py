import torch
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        # >>> Initialize a meter to record loss
        # >>>     losses = AverageMeter()
        # >>> Update meter after every minibatch update
        # >>>     losses.update(loss_value, batch_size)
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


def accuracy(logit, target, top_k=(1,), ignore_idx=255):
    """ Suppose you have the ground truth prediction tensor y of shape b-h-w (dtype=torch.int64).
        Your modules predict per-pixel class logit of shape b-c-h-w, with c is the number of
        classes (including "background"). These logit are the "raw" predictions before softmax
        function transforms them into class probabilities. Since we are only looking at the top k,
        it does not matter if the predictions are "raw" or "probabilities".
    """
    max_k = max(top_k)

    # compute the top k predicted classes, per pixel
    _, tk = torch.topk(logit, max_k, dim=1)

    # you now have k predictions per pixel, and you want that one of them
    # will match the true labels target
    correct_pixels = torch.eq(target[:, None, ...], tk).any(dim=1).float()

    # take the mean of correct_pixels to get the overall average top-k accuracy
    valid = target != ignore_idx
    top_k_acc = correct_pixels[valid].mean()

    return top_k_acc


def eval_image(predict, label, num_classes):
    index = np.where((label >= 0) & (label < num_classes))
    predict = predict[index]
    label = label[index]

    tp = np.zeros((num_classes, 1))
    fp = np.zeros((num_classes, 1))
    tn = np.zeros((num_classes, 1))
    fn = np.zeros((num_classes, 1))

    for i in range(0, num_classes):
        tp[i] = np.sum(label[np.where(predict == i)] == i)
        fp[i] = np.sum(label[np.where(predict == i)] != i)
        tn[i] = np.sum(label[np.where(predict != i)] != i)
        fn[i] = np.sum(label[np.where(predict != i)] == i)

    return tp, fp, tn, fn, len(label)
