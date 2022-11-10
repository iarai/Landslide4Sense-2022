import numpy as np


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
