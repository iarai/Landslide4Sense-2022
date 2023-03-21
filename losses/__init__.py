from modules.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

from modules.jaccard import JaccardLoss
from modules.dice import DiceLoss
from modules.focal import FocalLoss
from modules.lovasz import LovaszLoss
from modules.soft_bce import SoftBCEWithLogitsLoss
from modules.soft_ce import SoftCrossEntropyLoss
from modules.tversky import TverskyLoss
from modules.mcc import MCCLoss