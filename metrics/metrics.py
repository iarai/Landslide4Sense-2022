import torch
import numpy as np


# https://blog.csdn.net/weixin_43654653/article/details/107972147?ops_request_misc=&request_id=&biz_id=102&utm_term=pixel%20accuracy%E4%BB%A3%E7%A0%81&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-6-107972147.142%5Ev10%5Epc_search_result_control_group,157%5Ev8%5Econtrol&spm=1018.2226.3001.4187
# https://zhuanlan.zhihu.com/p/518797698

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        tn = self.confusion_matrix.sum() - (fp + fn + tp)
        return tp, fp, tn, fn

    def pixel_accuracy(self):
        """The ratio of correctly classified pixels to all pixels
        Returns:
            acc = (TP + TN) / (TP + TN + FP + TN)
        """
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        score = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        return score

    def pixel_accuracy_class(self):
        """Calculate the ratio of the correctly classified pixels of each class to
        all the pixels of that class and then calculate the average
        Returns:
            acc = TP / (TP + FP)
        """
        tp, fp, _, _ = self.get_tp_fp_tn_fn()
        score = tp / (tp + fp + 1e-7)
        return score

    def mean_intersection_over_union(self):
        """Calculate the IoU of each category and then average (Intersection and Union ratio).
        IoU is generally calculated based on classes, and it is also calculated on pictures;
        The IoU calculated based on the class is accumulated after the IoU calculation of each class,
        and then averaged, and the obtained is the global evaluation mIoU.
        Returns:
            mIoU = mean(tp / (tp + fp + fn))
        """
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        score = np.nanmean(tp / (tp + fp + fn + 1e-7))
        return score
    
    def jaccard_index(self):
        """Also known as the intersection_over_union.
        Returns:
            IoU = tp / (tp + fp + fn)
        """
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        score = tp / (tp + fp + fn + 1e-7)
        return score

    def frequency_weighted_intersection_over_union(self):
        """It is understood that the weighted summation of the iou of each category is performed according
        to the frequency of occurrence of each category.
        Returns:
            fwIOU = [(TP+FN)/(TP+FP+TN+FN)] * [TP / (TP + FP + FN)]
        """
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        freq = (tp + fn) / (tp + fp + tn + fn)
        iu = tp / (tp + fp + fn)
        score = (freq[freq > 0] * iu[freq > 0]).sum()
        return score

    def precision(self):
        tp, fp, _, _ = self.get_tp_fp_tn_fn()
        score = tp / (tp + fp + 1e-7)
        return score

    def recall(self):
        tp, _, _, fn = self.get_tp_fp_tn_fn()
        score = tp / (tp + fn + 1e-7)
        return score

    def f1(self):
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        score = (2.0 * precision * recall) / (precision + recall + 1e-7)
        return score
    
    def dice(self):
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        score = 2*tp / (2*tp + fp + fn + 1e-7)
        return score

    def mcc(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        num = tp*tn - fp*fn
        den = np.sqrt((tp+fn)*(tp+fp)*(tn+fn)*(tn+fp))
        score = num / (den + 1e-7)
        return score
    
    def sensitivity(self):
        tp, _, _, fn = self.get_tp_fp_tn_fn()
        score = tp / (tp + fn + 1e-7)
        return score

    def specificity(self):
        _, fp, tn, _ = self.get_tp_fp_tn_fn()
        score = tn / (tn + fp + 1e-7)
        return score
    
    def fbeta(self, beta=1):
        self.beta = beta

        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        beta_tp = (1 + self.beta**2) * tp
        beta_fn = (self.beta**2) * fn
        score = beta_tp / (beta_tp + beta_fn + fp + 1e-7)
        return score

    def balanced_accuracy(self):
        score = (self.sensitivity() + self.specificity()) / 2
        return score

    def positive_predictive_value(self):
        tp, fp, _, _ = self.get_tp_fp_tn_fn()
        score = tp / (tp + fp + 1e-7)
        return score

    def negative_predictive_value(self):
        _, _, tn, fn = self.get_tp_fp_tn_fn()
        score = tn / (tn + fn + 1e-7)
        return score

    def false_negative_rate(self):
        tp, _, _, fn = self.get_tp_fp_tn_fn()
        return fn / (fn + tp)

    def false_positive_rate(self):
        _, fp, tn, _ = self.get_tp_fp_tn_fn()
        score = fp / (fp + tn + 1e-7)
        return score

    def false_discovery_rate(self):
        score = 1 - self.positive_predictive_value()
        return score

    def false_omission_rate(self):
        score = 1 - self.negative_predictive_value()
        return score

    def positive_likelihood_ratio(self):
        socre = self.sensitivity() / (self.false_positive_rate() + 1e-7)
        return socre

    def negative_likelihood_ratio(self):
        score = self.false_negative_rate() / (self.specificity())
        return score    

    def _generate_matrix(self, gt_image, pre_image):
        """confusion matrix
        Args:
            gt_image: Segmentation map label (numpy array form)
            pre_image: Prediction segmentation map (numpy array format)
        Returns:
        """
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        for lp, lt in zip(pre_image, gt_image):
            self.confusion_matrix += self._generate_matrix(
                lt.flatten(), lp.flatten())

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


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
