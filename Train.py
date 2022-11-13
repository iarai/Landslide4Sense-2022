import argparse
import numpy as np
import time
import os
import copy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import *
from dataset.landslide_dataset import LandslideDataSet
import importlib
from dataset.kfold import get_train_test_list, kfold_split

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

name_classes = ['Non-Landslide', 'Landslide']
epsilon = 1e-14


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


def importName(modulename, name):
    """
    Import a named object from a module in the context of this function.
    """
    try:
        module = __import__(modulename, globals(), locals(), [name])
    except ImportError:
        return None
    return vars(module)[name]


def get_arguments():
    parser = argparse.ArgumentParser(description="Baseline method for Land4Seen")

    parser.add_argument("--data_dir", type=str, default='/scratch/Land4Sense_Competition_h5/',
                        help="dataset path.")
    parser.add_argument("--model_module", type=str, default='model.Networks',
                        help='model module to import')
    parser.add_argument("--model_name", type=str, default='unet',
                        help='model name in given module')
    parser.add_argument("--train_list", type=str, default='./dataset/train.txt',
                        help="training list file.")
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt',
                        help="test list file.")
    parser.add_argument("--input_size", type=str, default='128,128',
                        help="width and height of input images.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for multithread data-loading.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="learning rate.")
    parser.add_argument("--num_steps", type=int, default=5000,
                        help="number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=5000,
                        help="number of training steps for early stopping.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="gpu id in the training.")
    parser.add_argument("--snapshot_dir", type=str, default='./exp/',
                        help="where to save snapshots of the model.")
    parser.add_argument("--kfold", type=int, default=10,
                        help="number of fold for kfold.")

    return parser.parse_args()


def main():
    # Namespace(batch_size=32, data_dir='data/', gpu_id=0, input_size='128,128', learning_rate=0.001,
    #           model_module='model.Networks', model_name='unet', num_classes=2, num_steps=5000, num_steps_stop=5000,
    #           num_workers=4, snapshot_dir='./exp/', test_list='./dataset/train.txt', train_list='./dataset/train.txt',
    #           weight_decay=0.0005)

    args = get_arguments()

    # specify which GPU(s) to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # get size of images (128, 128)
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    # enables cudnn for some operations such as conv layers and RNNs, which can yield a significant speedup.
    cudnn.enabled = True

    # set True to speed up constant image size inference
    cudnn.benchmark = True

    # Spliting k-fold
    kfold_split(num_fold=args.kfold, test_image_number=int(get_size_dataset() / args.kfold))

    # Create network
    model_import = importName(args.model_module, args.model_name)  # <class 'model.Networks.unet'>
    model = model_import(n_classes=args.num_classes)  # return model structure

    for fold in range(args.kfold):
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
        model_.train()

        # send your model to the "current device"
        model_ = model_.cuda()

        # <torch.utils.data.dataloader.DataLoader object at 0x7fa2ff5af390>
        src_loader = data.DataLoader(LandslideDataSet(args.data_dir, args.train_list,
                                                      max_iters=args.num_steps_stop * args.batch_size, set='labeled'),
                                     batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=True)

        # <torch.utils.data.dataloader.DataLoader object at 0x7f780a0537d0>
        test_loader = data.DataLoader(LandslideDataSet(args.data_dir, args.test_list, set='labeled'),
                                      batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        optimizer = optim.Adam(model_.parameters(),
                               lr=args.learning_rate, weight_decay=args.weight_decay)

        # resize picture
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

        # Dung de luu ket qua dat duoc qua num_steps_stop lan train: timem cross_entropy_loss_value, batch_oa
        # Example: Time: 10.44 Batch_OA = 95.7 cross_entropy_loss = 0.329
        hist = np.zeros((args.num_steps_stop, 3))

        # Dung de so sanh va luu cac trong so khi F1 > F1_best
        train_loss_best = 5.0

        # computes the cross entropy loss between input logits and target. the dataset background label is 255,
        # so we ignore the background when calculating the cross entropy
        cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255)

        # batch_id: 0 -->  num_steps_stop
        # src_data --> data
        for batch_id, src_data in enumerate(src_loader):
            if batch_id == args.num_steps_stop:
                break

            tem_time = time.time()

            # send your model to the "current device"
            model_.train()

            # Sets gradients of all model parameters to zero.
            # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            optimizer.zero_grad()

            # src_data return: image, label, np.array(size)=[14, 128, 128], name="picture name"
            images, labels, _, _ = src_data

            # Put the tensor in cuda
            images = images.cuda()

            # This runs the image through the network and gets a prediction for the object in the image.
            pred = model_(images)

            # resize pred results to network's input size
            pred_interp = interp(pred)

            # CE Loss
            labels = labels.cuda().long()  # GPU tensor - torch.cuda.LongTensor
            cross_entropy_loss_value = cross_entropy_loss(pred_interp, labels)
            _, predict_labels = torch.max(pred_interp, 1)
            predict_labels = predict_labels.detach().cpu().numpy()
            labels = labels.cpu().numpy()
            batch_oa = np.sum(predict_labels == labels) * 1. / len(labels.reshape(-1))

            hist[batch_id, 0] = cross_entropy_loss_value.item()
            hist[batch_id, 1] = batch_oa

            cross_entropy_loss_value.backward()
            optimizer.step()

            hist[batch_id, -1] = time.time() - tem_time

            if (batch_id + 1) % 100 == 0:
                print('Iter %d/%d Time: %.2f cross_entropy_loss = %.3f' % (batch_id + 1, args.num_steps,
                                                                           10 * np.mean(
                                                                               hist[batch_id - 9:batch_id + 1, -1]),
                                                                           np.mean(hist[batch_id - 9:batch_id + 1, 0])))

                if np.mean(hist[batch_id - 9:batch_id + 1, 0]) < train_loss_best:
                    train_loss_best = np.mean(hist[batch_id - 9:batch_id + 1, 0])
                    torch.save(model_.state_dict(), os.path.join(snapshot_dir, 'model_weight.pth'))

        # Later to restore:
        model_.load_state_dict(torch.load(os.path.join(snapshot_dir, 'model_weight.pth')))
        model_.eval()
        TP_all = np.zeros((args.num_classes, 1))
        FP_all = np.zeros((args.num_classes, 1))
        TN_all = np.zeros((args.num_classes, 1))
        FN_all = np.zeros((args.num_classes, 1))
        # n_valid_sample_all = 0
        P = np.zeros((args.num_classes, 1))
        R = np.zeros((args.num_classes, 1))
        F1 = np.zeros((args.num_classes, 1))
        Acc = np.zeros((args.num_classes, 1))
        Spec = np.zeros((args.num_classes, 1))

        # y_true_all = []
        # y_pred_all = []

        for _, batch in enumerate(test_loader):
            image, label, _, name = batch
            label = label.squeeze().numpy()
            image = image.float().cuda()

            with torch.no_grad():
                pred = model_(image)

            _, pred = torch.max(interp(nn.functional.softmax(pred, dim=1)).detach(), 1)
            pred = pred.squeeze().data.cpu().numpy()

            # Return TP, FP, TN, FN for each batch
            TP, FP, TN, FN, _ = eval_image(pred.reshape(-1), label.reshape(-1), args.num_classes)

            # Calculating for all of batch
            TP_all += TP
            FP_all += FP
            TN_all += TN
            FN_all += FN
            # n_valid_sample_all += n_valid_sample

            # y_true_all.extend(label.reshape(-1))
            # y_pred_all.extend(pred.reshape(-1))

        # OA = np.sum(TP_all) * 1.0 / n_valid_sample_all
        for i in range(args.num_classes):
            P[i] = TP_all[i] * 1.0 / (TP_all[i] + FP_all[i] + epsilon)
            R[i] = TP_all[i] * 1.0 / (TP_all[i] + FN_all[i] + epsilon)
            Acc[i] = (TP_all[i] + TN_all[i]) / (TP_all[i] + TN_all[i] + FP_all[i] + FN_all[i])
            Spec[i] = TN_all[i] / (TN_all[i] + FP_all[i])
            F1[i] = 2.0 * P[i] * R[i] / (P[i] + R[i] + epsilon)
            # if i == 1:
            # print('===>' + name_classes[i] + ' Precision: %.2f' % (P * 100))
            # print('===>' + name_classes[i] + ' Recall: %.2f' % (R * 100))
            # print('===>' + name_classes[i] + ' F1: %.2f' % (F1[i] * 100))

        print('===> Non-Acc = %.2f Non-Pre = %.2f Non-Rec = %.2f Non-Spec = %.2f Non-F1 = %.2f Non-TP = %d Non-TN = %d Non-FP = %d Non-FN = %d' %
              (Acc[0] * 100, P[0] * 100, R[0] * 100, Spec[0] * 100, F1[0] * 100, TP_all[0], TN_all[0], FP_all[0], FN_all[0]))

        print('===> Land-Acc = %.2f Land-Pre = %.2f Land-Rec = %.2f Land-Spec = %.2f Land-F1 = %.2f Land-TP = %d Land-TN = %d Land-FP = %d Land-FN = %d' %
              (Acc[1] * 100, P[1] * 100, R[1] * 100, Spec[1] * 100, F1[1] * 100, TP_all[1], TN_all[1], FP_all[1], FN_all[1]))

        print('===> Mean-Acc = %.2f Mean-Pre = %.2f Mean-Rec = %.2f Mean-Spec = %.2f Mean-F1 = %.2f' %
              (np.mean(Acc) * 100, np.mean(P) * 100, np.mean(R) * 100, np.mean(Spec) * 100, np.mean(F1) * 100))

        # cm = confusion_matrix([int(x) for x in y_true_all], [int(x) for x in y_pred_all], labels=name_classes)
        # plt.figure(figsize=(12.8, 6))
        # sns.heatmap(cm, annot=True, xticklabels=name_classes, yticklabels=name_classes, cmap="Blues", fmt="g")
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.title('Confusion Matrix')


if __name__ == '__main__':
    main()
