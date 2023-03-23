from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import copy as cp
import torch
import numpy as np

from torch.utils import data
import albumentations as A
from torch.autograd import Variable

from metrics.metrics import *
from utils import Kpar
from utils.helpers import import_name, get_size_dataset, split_fold, get_train_test_list
from utils.saver import Saver
from dataset.dataset import LandslideDataSet
from losses import dice, focal, jaccard, lovasz, mcc, soft_bce, soft_ce, tversky
from losses.hybrid import hybrid_loss

# from modules.UNet import UNet_CBAM


def parse_args():
    """Parse all the arguments provided from the CLI.

        Returns:
          A list of parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Train a Semantic Segmentation network")

    parser.add_argument("--model_module", type=str, default='modules',
                        help='model module to import')
    parser.add_argument("--model_name", type=str, default='ResUNet_2Plus',
                        help='model name in given module: UNet, UNet_Att, UNet_2Plus, UNet_2Plus_CBAM, UNet_3Plus, UNet_3Plus_CBAM, ResUNet_2Plus')

    parser.add_argument("--data_dir", type=str, default='./TrainData/',
                        help="dataset path.")
    parser.add_argument("--train_list", type=str, default='./dataset/train.txt',
                        help="training list file.")
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt',
                        help="test list file.")
    parser.add_argument('--dataset', dest='dataset', type=str, default='Landslide4Sense',
                        help='training dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="number of images sent to the network in one step.")
    parser.add_argument('--start_epoch', dest='start_epoch', type=int, default=0,
                        help='starting epoch')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread data-loading.")

    # cuda
    parser.add_argument('--cuda', dest='cuda', type=bool, default=True,
                        help='whether use CUDA')

    # multiple GPUs
    parser.add_argument('--mGPUs', dest='mGPUs', type=bool, default=False,
                        help='whether use multiple GPUs')
    parser.add_argument('--gpu_ids', dest='gpu_ids', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')

    parser.add_argument("--save_dir", type=str, default='./exp/',
                        help="where to save snapshots of the modules.")

    parser.add_argument("--k_fold", type=int, default=10,
                        help="number of fold for k-fold.")

    # config optimization
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='adamax',
                        help='training optimizer: adam, adamax, adamW, sgd')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3,
                        help='starting learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5,
                        help='weight_decay')
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', type=int, default=50,
                        help='step to do learning rate decay, uint is epoch')
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', type=float, default=1e-1,
                        help='learning rate decay ratio')

    # set training session
    parser.add_argument('--s', dest='session', type=int, default=1,
                        help='training session')

    # resume trained model
    parser.add_argument('--r', dest='resume', type=bool, default=False,
                        help='resume checkpoint or not')
    parser.add_argument('--checksession', dest='checksession', type=int, default=1,
                        help='checksession to load model')
    parser.add_argument('--checkepoch', dest='checkepoch', type=int, default=1,
                        help='checkepoch to load model')
    parser.add_argument('--checkpoint', dest='checkpoint', type=int, default=0,
                        help='checkpoint to load model')

    parser.add_argument('--checkname', dest='checkname', type=str, default=None,
                        help='checkname')

    # configure validation
    parser.add_argument('--no_val', dest='no_val', type=bool, default=False,
                        help='not do validation')

    parser.add_argument('--loss_func', dest='loss_func', type=str, default='dice',
                        help='loss function')

    return parser.parse_args()


def adjust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


train_transform = A.Compose(
    [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.ShiftScaleRotate(),
        A.CoarseDropout(),
        A.MaskDropout(),
        A.PixelDropout(),
        A.Rotate(),
    ]
)


def get_loss_function(args):
    if args.loss_func == 'dice':
        return dice.DiceLoss('multiclass')
    elif args.loss_func == 'focal':
        return focal.FocalLoss('multiclass')
    elif args.loss_func == 'jaccard':
        return jaccard.JaccardLoss('multiclass')
    elif args.loss_func == 'lovasz':
        return lovasz.LovaszLoss('multiclass')
    elif args.loss_func == 'mcc':
        return mcc.MCCLoss
    elif args.loss_func == 'soft_bce':
        return soft_bce.SoftBCEWithLogitsLoss
    elif args.loss_func == 'soft_ce':
        return soft_ce.SoftCrossEntropyLoss
    elif args.loss_func == 'tversky':
        return tversky.TverskyLoss('muticlass')
    elif args.loss_func == 'hybrid':
        return hybrid_loss
    else:
        raise ValueError("Choice of loss function")


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(self.args)
        self.saver.save_experiment_config()

        # Define Dataloader
        self.train_loader = data.DataLoader(LandslideDataSet(args.data_dir, args.train_list, transform=None, set_mask='masked'),
                                            batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers, pin_memory=True)

        self.test_loader = data.DataLoader(LandslideDataSet(args.data_dir, args.test_list, set_mask='masked'),
                                           batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True)

        # Define network
        model_import = import_name(self.args.model_module, self.args.model_name)
        model = model_import(n_classes=self.args.num_classes)        

        # Define Optimizer
        self.lr = self.args.lr

        if args.optimizer == 'adam':
            self.lr = self.lr * 0.1
            opt = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamax':
            self.lr = self.lr * 0.1
            opt = torch.optim.Adamax(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamW':
            self.lr = self.lr * 0.1
            opt = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            opt = torch.optim.SGD(
                model.parameters(), lr=args.lr, momentum=0, weight_decay=args.weight_decay)

        # Define criterion
        self.criterion = get_loss_function(self.args)

        self.model = model
        self.optimizer = opt

        # Define Evaluator
        self.evaluator = Evaluator(self.args.num_classes)

        # multiple mGPUs
        if self.args.mGPUs:
            self.model = torch.nn.DataParallel(
                self.model, device_ids=self.args.gpu_ids)

        # Using cuda
        if self.args.cuda:
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.f1_best_pred = -np.Inf
        self.lr_stage = [20, 40, 60, 80]
        self.lr_stage_ind = 0

    def training(self, epoch, kbar):
        train_loss = 0.0
        self.model.train()

        if self.lr_stage_ind > 1 and epoch % (self.lr_stage[self.lr_stage_ind]) == 0:
            adjust_learning_rate(self.optimizer, self.args.lr_decay_gamma)
            self.lr *= self.args.lr_decay_gamma
            self.lr_stage_ind += 1

        for batch_id, batch in enumerate(self.train_loader):
            image, target, _, _ = batch

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.optimizer.zero_grad()

            inputs = Variable(image)
            labels = Variable(target)

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels.long())
            loss.backward(torch.ones_like(loss))
            self.optimizer.step()
            train_loss += loss.item()

            kbar.update(batch_id, values=[("loss", train_loss)])

        # save checkpoint every epoch
        if self.args.no_val:
            is_best = False

            self.saver.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred
                }, is_best)

    def validation(self, epoch, kbar):
        self.model.eval()
        self.evaluator.reset()
        val_loss = 0.0

        for _, batch in enumerate(self.test_loader):
            image, target, _, _ = batch

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(image)

            loss = self.criterion(output, target.long())
            val_loss += loss.item()
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        acc = self.evaluator.pixel_accuracy()
        mIoU = self.evaluator.mean_intersection_over_union()
        sen = self.evaluator.sensitivity()
        spec = self.evaluator.specificity()
        pre = self.evaluator.precision()
        rec = self.evaluator.recall()
        f1 = self.evaluator.f1()
        dice = self.evaluator.dice()
        jac = self.evaluator.jaccard_index()

        kbar.add(1, values=[("val_loss", val_loss), ("Acc", acc[1]), ("mIoU", mIoU),
                            ('sensitivity', sen[1]),
                            ('specificity', spec[1]), ('dice',
                                                       dice[1]), ('jaccard', jac[1]),
                            ('precision', pre[1]), ('recall', rec[1]), ('f1', f1[1])])

        new_f1_pred = f1[1]

        if new_f1_pred > self.f1_best_pred:
            print('\nEpoch %d: f1 improved from %0.5f to %0.5f' % (
                epoch + 1, self.f1_best_pred, new_f1_pred))

            is_best = True
            self.f1_best_pred = new_f1_pred
            self.saver.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.f1_best_pred
                }, is_best)
        else:
            print('\nEpoch %d: f1 (%.05f) did not improve from %0.5f' %
                  (epoch + 1, new_f1_pred, self.f1_best_pred))


def main():
    args = parse_args()

    if args.save_dir is None:
        args.save_dir = os.path.join(os.getcwd(), 'run')

    if args.checkname is None:
        args.checkname = 'model_' + str(args.model_name)

    if args.cuda and args.mGPUs:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError(
                'Argument --gpu_ids must be a comma-separated list of integer only')

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.lr is None:
        lrs = {'Landslide4Sense': 0.01}
        args.lr = lrs[args.dataset.lower()] / \
            (4 * len(args.gpu_ids)) * args.batch_size

    # Splitting k-fold
    # split_fold(num_fold=args.k_fold, test_image_number=int(
    #     get_size_dataset('./data/img') / args.k_fold))

    for fold in range(args.k_fold):
        print("\nTraining on fold %d" % fold)

        # Creating train.txt and test.txt
        get_train_test_list(fold)

        # create snapshots directory
        snapshot_dir = args.save_dir + "fold" + str(fold)
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        trainer = Trainer(args)

        # Takes a local copy of the machine learning algorithm (modules) to avoid changing the one passed in
        trainer_ = cp.deepcopy(trainer)

        train_per_epoch = np.ceil(get_size_dataset(
            "./data/TrainData" + str(fold) + "/train/img/") / args.batch_size)

        for epoch in range(trainer_.args.start_epoch, trainer_.args.epochs):
            kbar = Kpar.Kbar(target=train_per_epoch, epoch=epoch,
                             num_epochs=args.epochs, width=25, always_stateful=False)

            trainer_.training(epoch, kbar)
            trainer_.validation(epoch, kbar)


if __name__ == '__main__':
    main()
