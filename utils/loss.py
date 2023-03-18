import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.lovasz_losses as L


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.batch_average = batch_average
        self.cuda = cuda

        if size_average:
            self.size_average = 'mean'
        else:
            self.size_average = 'sum'

    def build_loss(self, mode='ce'):
        """Choices: ['ce', 'focal', 'ls' or 'lh']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'ls':
            return self.LovaszLossSoftmax
        elif mode == 'lh':
            return self.LovaszLossHinge
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def LovaszLossSoftmax(self, logit, target):
        n, c, h, w = logit.size()
        out = F.softmax(logit, dim=1)
        loss = L.lovasz_softmax(out, target)

        if self.batch_average:
            loss /= n

        return loss

    def LovaszLossHinge(self, logit, target):        
        n, c, h, w = logit.size()
        loss = L.lovasz_hinge(logit, target)

        if self.batch_average:
            loss /= n

        return loss



if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
