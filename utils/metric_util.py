import numpy as np
from torchmetrics import Metric
from dataloader.pc_dataset import get_SemKITTI_label_name
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
import time


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist


class IoU(Metric):
    def __init__(self, dataset_config, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.hist_list = []
        self.best_miou = 0
        self.SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
        self.unique_label = np.asarray(sorted(list(self.SemKITTI_label_name.keys())))[1:] - 1
        self.unique_label_str = [self.SemKITTI_label_name[x] for x in self.unique_label + 1]

    def update(self, predict_labels, val_pt_labs) -> None:
        self.hist_list.append(fast_hist_crop(predict_labels, val_pt_labs, self.unique_label))

    def compute(self):
        iou = per_class_iu(sum(self.hist_list))
        if np.nanmean(iou) > self.best_miou:
            self.best_miou = np.nanmean(iou)
        self.hist_list = []
        return iou, self.best_miou

class OoD_e(Metric):
    def __init__(self, dataset_config, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.hist_list_scores = []
        self.hist_list_labels = []
        self.dataset_config = dataset_config

    def update(self, uncer_scores, val_pt_labs) -> None:
        self.hist_list_scores.append(uncer_scores)
        self.hist_list_labels.append(val_pt_labs)

    def compute(self):
        uncer_scores_all = np.concatenate(self.hist_list_scores, axis=-1)
        labels_all = np.concatenate(self.hist_list_labels, axis=-1)

        if 'kitti' in self.dataset_config['dataset_type']:
            uncer_scores_all = uncer_scores_all[labels_all != 0]
            labels_all = labels_all[labels_all != 0]
            for i in ([1, 52, 99]):
                labels_all[labels_all == i] = -1
            
            labels_all[labels_all != -1] = 0
            labels_all[labels_all == -1] = 1

        else:
            for i in ([0, 31]):
                uncer_scores_all = uncer_scores_all[labels_all != i]
                labels_all = labels_all[labels_all != i]
            for i in ([1,5,7,8,10,11,13,19,20,29]):
                labels_all[labels_all == i] = -1
            labels_all[labels_all != -1] = 0
            labels_all[labels_all == -1] = 1

        precision, recall, _ = precision_recall_curve(labels_all, uncer_scores_all)
        aupr_score = auc(recall, precision)

        fpr, tpr, _ = roc_curve(labels_all, uncer_scores_all)
        auroc_score = auc(fpr, tpr)

        fpr95 = fpr[tpr > 0.95][0]

        print('AUROC: ', auroc_score, 'AUPR: ', aupr_score, 'FPR95: ', fpr95)

        # return labels_all, uncer_scores_all