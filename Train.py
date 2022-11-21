import argparse
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
from dataset.kfold import get_train_test_list, kfold_split

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import Deform_CNN.arch as archs

arch_names = archs.__dict__.keys()

name_classes = ['Non-Landslide', 'Landslide']
epsilon = 1e-14


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='LandSlide_Net',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DeformCNN', choices=arch_names,
                        help='model architecture: ' + ' | '.join(arch_names) + ' (default: Dcnv2LandSlideNet)')

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
                        help="width and height of input images.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="number of images in each batch.")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="learning rate.")
    parser.add_argument("--num_steps", type=int, default=10000,
                        help="number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=10000,
                        help="number of training steps for early stopping.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--data_dir", type=str, default='./TrainData/',
                        help="dataset path.")
    parser.add_argument("--train_list", type=str, default='./dataset/train.txt',
                        help="training list file.")
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt',
                        help="test list file.")
    parser.add_argument("--snapshot_dir", type=str, default='./exp/',
                        help="where to save snapshots of the model.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="number of workers for multithread data-loading.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="gpu id in the training.")
    parser.add_argument("--k_fold", type=int, default=10,
                        help="number of fold for k-fold.")

    args = parser.parse_args()

    return args


def main():
    args = get_arguments()

    if args.name is None:
        args.name = '%s' % args.arch
        if args.deform:
            args.name += '_wDCN_1fc'
            if args.modulation:
                args.name += 'v2'
            args.name += '_dcn-%d-' % args.dcn
            args.name += 'cvn-%d' % args.cvn

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # get size of images (128, 128)
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    # enables cudnn for some operations such as conv layers and RNNs, which can yield a significant speedup.
    cudnn.enabled = True

    # set True to speed up constant image size inference
    cudnn.benchmark = True

    # Spliting k-fold
    kfold_split(num_fold=args.k_fold, test_image_number=int(get_size_dataset() / args.k_fold))

    # create model
    model = archs.__dict__[args.arch](args, args.num_classes)

    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    Pre_classes = []
    Rec_classes = []
    F1_classes = []
    Acc_classes = []
    Spec_classes = []

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

        optimizer = optim.Adagrad(model_.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-09,
                                  weight_decay=args.weight_decay)

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, cycle_momentum=False,
                                                step_size_up=len(src_loader) * 8, mode='triangular2')

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

        # Here, we use enumerate(src_loader) instead of iter(src_loader)
        # so that we can track the batch index and do some intra-epoch reporting
        # batch_id: 0 -->  num_steps_stop
        # src_data --> data
        for batch_id, src_data in enumerate(src_loader):
            if batch_id == args.num_steps_stop:
                break

            tem_time = time.time()

            # send your model to the "current device"
            model_.train()

            # Sets gradients of all model parameters to zero for every batch!
            # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            optimizer.zero_grad()

            # Every data instance is an input + label pair
            # src_data return: image, label, np.array(size)=[14, 128, 128], name="picture name"
            images, labels, _, _ = src_data

            # Put the tensor in cuda
            images = images.cuda()

            # Make predictions for this batch
            pred = model_(images)

            # resize pred results to network's input size
            pred_interp = interp(pred)

            # CE Loss
            labels = labels.cuda().long()  # GPU tensor - torch.cuda.LongTensor

            # Compute the loss value
            cross_entropy_loss_value = cross_entropy_loss(pred_interp, labels)
            _, predict_labels = torch.max(pred_interp, 1)
            predict_labels = predict_labels.detach().cpu().numpy()
            labels = labels.cpu().numpy()
            batch_oa = np.sum(predict_labels == labels) * 1. / len(labels.reshape(-1))

            # Gather data and report
            hist[batch_id, 0] = cross_entropy_loss_value.item()
            hist[batch_id, 1] = batch_oa

            cross_entropy_loss_value.backward()  # compute gradient
            optimizer.step()  # Doing optimizing step (adjust learning weights)
            scheduler.step()

            # Gather data and report
            hist[batch_id, -1] = time.time() - tem_time

            # Reports the average per-batch loss for the last 100 batches
            if (batch_id + 1) % 100 == 0:
                print('Iter %d/%d Time: %.2f cross_entropy_loss = %.3f' %
                      (batch_id + 1, args.num_steps, 100 * np.mean(hist[batch_id - 99:batch_id + 1, -1]),
                       np.mean(hist[batch_id - 99:batch_id + 1, 0])))

                if np.mean(hist[batch_id - 99:batch_id + 1, 0]) < train_loss_best:
                    train_loss_best = np.mean(hist[batch_id - 99:batch_id + 1, 0])
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

        y_true_all = []
        y_pred_all = []

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

            y_true_all.append(label.reshape(-1))
            y_pred_all.append(pred.reshape(-1))

        # y_true_all = np.concatenate(y_true_all).tolist()
        # y_pred_all = np.concatenate(y_pred_all).tolist()

        actual_classes = np.append(actual_classes, np.concatenate(y_true_all).tolist())
        predicted_classes = np.append(predicted_classes, np.concatenate(y_pred_all).tolist())

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

        Pre_classes = np.append(Pre_classes, P)
        Rec_classes = np.append(Rec_classes, R)
        F1_classes = np.append(F1_classes, F1)
        Acc_classes = np.append(Acc_classes, Acc)
        Spec_classes = np.append(Spec_classes, Spec)

        print('----------------------------- For per fold ----------------------------------------')

        print(
            '===> Non-Landslide [Acc, Pre, Rec, Spec, F1, TP, TN, FP, FN] = [%.2f, %.2f, %.2f, %.2f, %.2f, %d, %d, %d, %d]' %
            (Acc[0] * 100, P[0] * 100, R[0] * 100, Spec[0] * 100, F1[0] * 100, TP_all[0], TN_all[0], FP_all[0],
             FN_all[0]))

        print(
            '===> Landslide [Acc, Pre, Rec, Spec, F1, TP, TN, FP, FN] = [%.2f, %.2f, %.2f, %.2f, %.2f, %d, %d, %d, %d]' %
            (Acc[1] * 100, P[1] * 100, R[1] * 100, Spec[1] * 100, F1[1] * 100, TP_all[1], TN_all[1], FP_all[1],
             FN_all[1]))

        print('===> Mean [Acc, Pre, Rec, Spec, F1] = [%.2f, %.2f, %.2f, %.2f, %.2f]' %
              (np.mean(Acc) * 100, np.mean(P) * 100, np.mean(R) * 100, np.mean(Spec) * 100, np.mean(F1) * 100))

        # cm = confusion_matrix(y_true_all, y_pred_all)
        # plt.figure(figsize=(12.8, 6))
        # sns.heatmap(cm, annot=True, xticklabels=name_classes, yticklabels=name_classes, cmap="Blues", fmt="g")
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.title('Confusion Matrix')
        # plt.savefig(os.path.join(snapshot_dir, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)
        # plt.close()

    print('----------------------------- For all folds ----------------------------------------')
    print(Acc_classes)
    print(np.mean(Acc_classes))

    print('===> Mean-Non-Landslide [Acc, Pre, Rec, Spec, F1] = [%.2f, %.2f, %.2f, %.2f, %.2f]' %
          (np.mean(Acc_classes[0:len(Acc_classes):2]) * 100, np.mean(Pre_classes[0:len(Pre_classes):2]) * 100,
           np.mean(Rec_classes[0:len(Rec_classes):2]) * 100, np.mean(Spec_classes[0:len(Spec_classes):2]) * 100,
           np.mean(F1_classes[0:len(F1_classes):2]) * 100))

    print('===> Mean-Landslide [Acc, Pre, Rec, Spec, F1] = [%.2f, %.2f, %.2f, %.2f, %.2f]' %
          (np.mean(Acc_classes[1:len(Acc_classes):2]) * 100, np.mean(Pre_classes[1:len(Pre_classes):2]) * 100,
           np.mean(Rec_classes[1:len(Rec_classes):2]) * 100, np.mean(Spec_classes[1:len(Spec_classes):2]) * 100,
           np.mean(F1_classes[1:len(F1_classes):2]) * 100))

    print('===> Mean [Acc, Pre, Rec, Spec, F1] = [%.2f, %.2f, %.2f, %.2f, %.2f]' %
          (np.mean(Acc_classes) * 100, np.mean(Pre_classes) * 100, np.mean(Rec_classes) * 100,
           np.mean(Spec_classes) * 100, np.mean(F1_classes) * 100))

    cm = confusion_matrix(actual_classes, predicted_classes)
    # cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    # sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=name_classes, yticklabels=name_classes, cmap="Blues")
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=name_classes, yticklabels=name_classes, cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join('/content/Landslide4Sense-2022', 'confusion_matrix.pdf'), bbox_inches='tight', dpi=2400)
    plt.close()


if __name__ == '__main__':
    main()
