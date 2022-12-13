import argparse
import time
import os
import copy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
import albumentations as A
import cv2
import random
from collections import OrderedDict

from utils.tools import *
from dataset.landslide_dataset import LandslideDataSet
from dataset.kfold import get_train_test_list, kfold_split

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import Deform_CNN.arch as archs

arch_names = archs.__dict__.keys()

name_classes = ['Non-Landslide', 'Landslide']
epsilon = 1e-14


def get_arguments():
    """Parse all the arguments provided from the CLI.

        Returns:
          A list of parsed arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='DeformCNN',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DeformCNN', choices=arch_names,
                        help='model architecture: ' + ' | '.join(arch_names) + ' (default: DeformCNN)')

    # deform False --> Use only regular convolution
    parser.add_argument('--deform', default=True, type=str2bool,
                        help='use deform conv')

    # modulation = True --> Use modulated deformable convolution at conv3~4
    # modulation = False --> use deformable convolution at conv3~4
    parser.add_argument('--modulation', default=True, type=str2bool,
                        help='use modulated deform conv')
    parser.add_argument('--dcn', default=4, type=int,
                        help='number of sub-layer')
    parser.add_argument('--cvn', default=2, type=int,
                        help='number of 1-D convolutions')
    parser.add_argument("--input_size", type=str, default='128,128',
                        help="comma-separated string with height and width of images.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="number of images sent to the network in one step.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4,
                        help="base learning rate for training with polynomial decay.")
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum component of the optimiser.")
    parser.add_argument("--data_dir", type=str, default='./TrainData/',
                        help="dataset path.")
    parser.add_argument("--train_list", type=str, default='./dataset/train.txt',
                        help="training list file.")
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt',
                        help="test list file.")
    parser.add_argument("--snapshot_dir", type=str, default='./exp/',
                        help="where to save snapshots of the model.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread data-loading.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="gpu id in the training.")
    parser.add_argument("--k_fold", type=int, default=10,
                        help="number of fold for k-fold.")
    return parser.parse_args()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, learning_rate, num_steps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


train_transform = A.Compose(
    [
        A.OneOf(
            [
                A.ShiftScaleRotate(shift_limit=0.2,
                                   scale_limit=0.2,
                                   rotate_limit=30,
                                   p=0.5),
                # A.OpticalDistortion(distort_limit=0.01,
                #                     shift_limit=0.1,
                #                     border_mode=cv2.BORDER_CONSTANT,
                #                     value=0,
                #                     p=0.5),
                # A.GridDistortion(num_steps=5,
                #                 border_mode=cv2.BORDER_CONSTANT,
                #                 value=0,
                #                 p=0.5)
            ], p=1.0),

        A.VerticalFlip(p=0.3),
        A.HorizontalFlip(p=0.3),
        A.Flip(p=0.5)
    ]
)


def train(args, train_loader, model, criterion, optimizer, epoch, interp):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    for batch_id, batch_data in enumerate(train_loader):
        image, label, _, _ = batch_data
        image = image.cuda()
        label = label.cuda().long()

        pred = interp(model(image))
        loss = criterion(pred, label)
        # acc = accuracy(pred, label)[0]

        losses.update(loss.item(), image.size(0))
        # scores.update(acc.item(), image.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        # ('acc', scores.avg),
    ])

    return log


#     for _, batch in enumerate(test_loader):
#         image, label, _, name = batch
#         label = label.squeeze().numpy()
#         image = image.float().cuda()

#         with torch.no_grad():
#             pred = model_(image)

#         _, pred = torch.max(interp(nn.functional.softmax(pred, dim=1)).detach(), 1)
#         pred = pred.squeeze().data.cpu().numpy()


def validate(args, val_loader, model, criterion, interp):
    losses = AverageMeter()
    scores = AverageMeter()

    model.eval()

    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            image, label, _, name = batch
            image = image.cuda()
            label = label.cuda().long()

            pred = interp(model(image))
            loss = criterion(pred, label)
            # acc = accuracy(pred, label)[0]

            losses.update(loss.item(), image.size(0))
            # scores.update(acc.item(), image.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        # ('acc', scores.avg),
    ])

    return log


def main():
    """Create the model and start the training."""
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # get size of images (128, 128)
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    # print('Config -----')
    # for arg in vars(args):
    #     print('%s: %s' % (arg, getattr(args, arg)))
    # print('------------')

    # enables cudnn for some operations such as conv layers and RNNs, which can yield a significant speedup.
    cudnn.enabled = True

    # set True to speed up constant image size inference
    cudnn.benchmark = True

    # Spliting k-fold
    kfold_split(num_fold=args.k_fold, test_image_number=int(get_size_dataset() / args.k_fold))

    # create model
    model = archs.__dict__[args.arch](args, args.num_classes)

    # actual_classes = np.empty([0], dtype=int)
    # predicted_classes = np.empty([0], dtype=int)
    # Pre_classes = []
    # Rec_classes = []
    # F1_classes = []
    # Acc_classes = []
    # Spec_classes = []

    for fold in range(args.k_fold):
        print("Training on Fold %d" % fold)

        # Creating train.txt and test.txt
        get_train_test_list(fold)

        # create snapshots directory
        snapshot_dir = args.snapshot_dir + "fold" + str(fold)
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        # Takes a local copy of the machine learning algorithm (model) to avoid changing the one passed in
        model_ = cp.deepcopy(model)

        # model.train() tells your model that you are training the model. This helps inform layers such as Dropout
        # and BatchNorm, which are designed to behave differently during training and evaluation. For instance,
        # in training mode, BatchNorm updates a moving average on each new batch;
        # whereas, for evaluation mode, these updates are frozen.
        # model_.train()

        # send your model to the "current device"
        model_ = model_.cuda(args.gpu_id)

        # resize picture
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)

        # computes the cross entropy loss between input logits and target. the dataset background label is 255,
        # so we ignore the background when calculating the cross entropy
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        # implement model.optim_parameters(args) to handle different models' lr setting
        # optimizer = optim.SGD(model_.parameters(args), lr=args.learning_rate, momentum=args.momentum,
        #                       weight_decay=args.weight_decay)

        optimizer = optim.Adam(model_.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-09,
                               weight_decay=args.weight_decay)

        # <torch.utils.data.dataloader.DataLoader object at 0x7fa2ff5af390>
        train_loader = data.DataLoader(LandslideDataSet(args.data_dir, args.train_list,
                                                        transform=None,
                                                        set_mask='masked'),
                                       batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True)

        # <torch.utils.data.dataloader.DataLoader object at 0x7f780a0537d0>
        test_loader = data.DataLoader(LandslideDataSet(args.data_dir, args.test_list, set_mask='masked'),
                                      batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                      pin_memory=True)

        # Dung de so sanh va luu cac trong so khi F1 > F1_best
        train_loss_best = 5.0

        for epoch in range(args.epochs):
            tem_time = time.time()

            # train for one epoch
            train_log = train(args, train_loader, model_, criterion, optimizer, epoch, interp)

            # evaluate on validation set
            val_log = validate(args, test_loader, model_, criterion, interp)

            # Gather data and report
            epoch_time = time.time() - tem_time

            # Reports the loss for the each epoch
            print('Epoch %d/%d - %.2fs - loss %.4f - val_loss %.4f' %
                  (epoch, args.epochs, epoch_time, train_log['loss'], val_log['loss']))

            # print('Epoch %d/%d - %.2fs - loss %.4f - acc %.4f - val_loss %.4f - val_acc %.4f' %
            #       (epoch, args.epochs, epoch_time, train_log['loss'], train_log['acc'], val_log['loss'], val_log['acc']))


if __name__ == '__main__':
    main()
