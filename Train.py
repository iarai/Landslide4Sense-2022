import argparse
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import *
from dataset.landslide_dataset import LandslideDataSet
import importlib
from dataset.kfold import get_train_test_list, kfold_split

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
    kfold_split(num_fold=args.kfold, test_image_number=int(get_size_dataset()/args.kfold))

    for fold in range(args.kfold):
        print("Training on Fold %d" % fold)

        # Creating train.txt and test.txt
        get_train_test_list(fold)

        # create snapshots directory
        snapshot_dir = args.snapshot_dir + "fold" + str(fold)
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        # Create network
        model_import = importName(args.model_module, args.model_name)  # <class 'model.Networks.unet'>
        model = model_import(n_classes=args.num_classes)  # return model structure

        # model.train() tells your model that you are training the model. This helps inform layers such as Dropout
        # and BatchNorm, which are designed to behave differently during training and evaluation. For instance,
        # in training mode, BatchNorm updates a moving average on each new batch;
        # whereas, for evaluation mode, these updates are frozen.
        model.train()

        # send your model to the "current device"
        model = model.cuda()

        # <torch.utils.data.dataloader.DataLoader object at 0x7fa2ff5af390>
        src_loader = data.DataLoader(LandslideDataSet(args.data_dir, args.train_list,
                                                      max_iters=args.num_steps_stop * args.batch_size, set='labeled'),
                                     batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=True)

        # <torch.utils.data.dataloader.DataLoader object at 0x7f780a0537d0>
        test_loader = data.DataLoader(LandslideDataSet(args.data_dir, args.test_list, set='labeled'),
                                      batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate, weight_decay=args.weight_decay)

        # resize picture
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

        # Dung de luu ket qua dat duoc qua num_steps_stop lan train: timem cross_entropy_loss_value, batch_oa
        # Example: Time: 10.44 Batch_OA = 95.7 cross_entropy_loss = 0.329
        hist = np.zeros((args.num_steps_stop, 3))

        # Dung de so sanh va luu cac trong so khi F1 > F1_best
        F1_best = 0.5

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
            model.train()

            # Sets gradients of all model parameters to zero.
            # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            optimizer.zero_grad()

            # src_data return: image, label, np.array(size)=[14, 128, 128], name="picture name"
            images, labels, _, _ = src_data

            # Put the tensor in cuda
            images = images.cuda()

            # This runs the image through the network and gets a prediction for the object in the image.
            pred = model(images)

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

            if (batch_id + 1) % 10 == 0:
                model.eval()
                TP_all = np.zeros((args.num_classes, 1))
                FP_all = np.zeros((args.num_classes, 1))
                TN_all = np.zeros((args.num_classes, 1))
                FN_all = np.zeros((args.num_classes, 1))
                n_valid_sample_all = 0
                F1 = np.zeros((args.num_classes, 1))

                for _, batch in enumerate(test_loader):
                    image, label, _, name = batch
                    label = label.squeeze().numpy()
                    image = image.float().cuda()

                    with torch.no_grad():
                        pred = model(image)

                    _, pred = torch.max(interp(nn.functional.softmax(pred, dim=1)).detach(), 1)
                    pred = pred.squeeze().data.cpu().numpy()

                    TP, FP, TN, FN, n_valid_sample = eval_image(pred.reshape(-1), label.reshape(-1), args.num_classes)
                    TP_all += TP
                    FP_all += FP
                    TN_all += TN
                    FN_all += FN
                    n_valid_sample_all += n_valid_sample

                OA = np.sum(TP_all) * 1.0 / n_valid_sample_all
                for i in range(args.num_classes):
                    P = TP_all[i] * 1.0 / (TP_all[i] + FP_all[i] + epsilon)
                    R = TP_all[i] * 1.0 / (TP_all[i] + FN_all[i] + epsilon)
                    F1[i] = 2.0 * P * R / (P + R + epsilon)

                    # # for landslide
                    # if i == 1:
                    #     print('===>' + name_classes[i] + ' Precision for landslide: %.2f' % (P * 100))
                    #     print('===>' + name_classes[i] + ' Recall for landslide: %.2f' % (R * 100))
                    #     print('===>' + name_classes[i] + ' F1 for landslide: %.2f' % (F1[i] * 100))

                # for both non-landslide and landslide
                mP = np.mean(P)
                mR = np.mean(R)
                mF1 = np.mean(F1)
                # print('===> mean F1 (both non-landslide and landslide: %.2f OA: %.2f' % (mF1 * 100, OA * 100))

                print(
                    'Iter %d/%d Time: %.2f Batch_OA = %.1f cross_entropy_loss = %.3f, mP = %.2f, mR = %.2f, mF = %.2f, OA = %.2f' %
                    (batch_id + 1, args.num_steps, 10 * np.mean(hist[batch_id - 9:batch_id + 1, -1]),
                     np.mean(hist[batch_id - 9:batch_id + 1, 1]) * 100, np.mean(hist[batch_id - 9:batch_id + 1, 0]),
                     mP * 100, mR * 100, mF1 * 100, OA * 100))

                if F1[1] > F1_best:
                    F1_best = F1[1]  # F1[1] --> calculate for landslide

                    # save the models
                    print('Save Model')
                    model_name = 'batch' + repr(batch_id + 1) + '_F1_' + repr(int(F1[1] * 10000)) + '.pth'
                    torch.save(model.state_dict(), os.path.join(
                        snapshot_dir, model_name))

            # # evaluation per 500 iterations
            # if (batch_id + 1) % 500 == 0:
            #     print('Testing..........')
            #     model.eval()
            #     TP_all = np.zeros((args.num_classes, 1))
            #     FP_all = np.zeros((args.num_classes, 1))
            #     TN_all = np.zeros((args.num_classes, 1))
            #     FN_all = np.zeros((args.num_classes, 1))
            #     n_valid_sample_all = 0
            #     F1 = np.zeros((args.num_classes, 1))

            #     for _, batch in enumerate(test_loader):
            #         image, label, _, name = batch
            #         label = label.squeeze().numpy()
            #         image = image.float().cuda()

            #         with torch.no_grad():
            #             pred = model(image)

            #         _, pred = torch.max(interp(nn.functional.softmax(pred, dim=1)).detach(), 1)
            #         pred = pred.squeeze().data.cpu().numpy()

            #         TP, FP, TN, FN, n_valid_sample = eval_image(pred.reshape(-1), label.reshape(-1), args.num_classes)
            #         TP_all += TP
            #         FP_all += FP
            #         TN_all += TN
            #         FN_all += FN
            #         n_valid_sample_all += n_valid_sample

            #     OA = np.sum(TP_all) * 1.0 / n_valid_sample_all
            #     for i in range(args.num_classes):
            #         P = TP_all[i] * 1.0 / (TP_all[i] + FP_all[i] + epsilon)
            #         R = TP_all[i] * 1.0 / (TP_all[i] + FN_all[i] + epsilon)
            #         F1[i] = 2.0 * P * R / (P + R + epsilon)

            #         # for landslide
            #         if i == 1:
            #             print('===>' + name_classes[i] + ' Precision for landslide: %.2f' % (P * 100))
            #             print('===>' + name_classes[i] + ' Recall for landslide: %.2f' % (R * 100))
            #             print('===>' + name_classes[i] + ' F1 for landslide: %.2f' % (F1[i] * 100))

            #     # for both non-landslide and landslide
            #     mF1 = np.mean(F1)
            #     print('===> mean F1 (both non-landslide and landslide: %.2f OA: %.2f' % (mF1 * 100, OA * 100))

            #     if F1[1] > F1_best:
            #         F1_best = F1[1]     # F1[1] --> calculate for landslide

            #         # save the models
            #         print('Save Model')
            #         model_name = 'batch' + repr(batch_id + 1) + '_F1_' + repr(int(F1[1] * 10000)) + '.pth'
            #         torch.save(model.state_dict(), os.path.join(
            #             snapshot_dir, model_name))


if __name__ == '__main__':
    main()
