from .dice import DiceLoss
from .focal import FocalLoss


def hybrid_loss(prediction, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(mode='multiclass', gamma=0, alpha=None)
    dice = DiceLoss(mode='multiclass')

    bce = focal(prediction, target)
    dce = dice(prediction, target)
    loss += bce + dce

    return loss

