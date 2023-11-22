import os
import argparse
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
from sklearn import manifold

def parse_args():
    '''Command instruction:
        source activate mmaction
        python experiments/compare_openness.py --ind_ncls 101 --ood_ncls 51
    '''
    parser = argparse.ArgumentParser(description='Compare the performance of openness')
    # model config
    parser.add_argument('--base_model', default='i3d', help='the backbone model name')
    parser.add_argument('--baselines', nargs='+', default=['I3D_Dropout_BALD', 'I3D_BNN_BALD', 'I3D_EDLlog_EDL', 'I3D_EDLlogAvUC_EDL'])
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.000423, 0.000024, 0.495783, 0.495783])
    parser.add_argument('--styles', nargs='+', default=['-b', '-k', '-r', '-g', '-m'])
    parser.add_argument('--ind_ncls', type=int, default=101, help='the number of classes in known dataset')
    parser.add_argument('--ood_ncls', type=int, help='the number of classes in unknwon dataset')
    parser.add_argument('--ood_data', default='HMDB', help='the name of OOD dataset.')
    parser.add_argument('--num_rand', type=int, default=10, help='the number of random selection for ood classes')
    parser.add_argument('--result_png', default='F1_openness_compare_HMDB.png')
    parser.add_argument('--t_SNE', default=False, help='plot the embedding')
    parser.add_argument('--t_SNE_cls', default=False, help='plot the embedding of an ind class')
    parser.add_argument('--analyze', default=False, help="analyze score distribution")
    args = parser.parse_args()
    return args


def main():

    result_file = "/mnt/data_nas/jcenaa/code_av/2DPASS/ckpt/lidar_maxlogits.npz"
    result_file = "/mnt/data_nas/jcenaa/code_av/2DPASS/ckpt/fusion_maxlogit_RCF_2gpu_lidar.npz"
    png_file = "/mnt/data_nas/jcenaa/code_av/2DPASS/ckpt/images"
    fontsize = 15
    assert os.path.exists(result_file), "File not found! Run ood_detection first!"
    # load the testing results
    results = np.load(result_file, allow_pickle=True)
    predictions = results['predictions'].squeeze()  # (N1,)
    labels = results['labels'].squeeze()  # (N2,)
    uncertainty = results['uncertainty'].squeeze()  # (N1,)
    labels_raw = results['labels_raw'].squeeze()  # (N2,)

    predictions = predictions[labels_raw != 0]
    labels = labels[labels_raw != 0]
    uncertainty = uncertainty[labels_raw != 0]
    labels_raw = labels_raw[labels_raw != 0]

    for i in ([1, 52, 99]):
        labels_raw[labels_raw == i] = -1
    labels_raw[labels_raw != -1] = 0
    labels_raw[labels_raw == -1] = 1

    ind_uncertainties_correct = uncertainty[(labels_raw == 0) * (predictions == labels)]
    ind_uncertainties_wrong = uncertainty[(labels_raw == 0) * (predictions != labels)]
    ood_uncertainties = uncertainty[labels_raw == 1]

    uncertainties = np.concatenate((ind_uncertainties_correct, ind_uncertainties_wrong, ood_uncertainties), axis=0)
    labels = np.concatenate((np.zeros_like(ind_uncertainties_correct), np.ones_like(ind_uncertainties_wrong), np.ones_like(ood_uncertainties)), axis=0)
    auroc = roc_auc_score(labels, uncertainties)
    aupr = average_precision_score(labels, uncertainties)
    fpr, tpr, _ = roc_curve(labels, uncertainties)
    fpr95 = fpr[tpr > 0.95][0]
    print(auroc, aupr, fpr95)

    ind_uncertainties_correct = (ind_uncertainties_correct - np.min(uncertainties)) / (np.max(uncertainties) - np.min(uncertainties))
    ind_uncertainties_wrong = (ind_uncertainties_wrong - np.min(uncertainties)) / (np.max(uncertainties) - np.min(uncertainties))
    ood_uncertainties = (ood_uncertainties - np.min(uncertainties)) / (np.max(uncertainties) - np.min(uncertainties))

    plt.figure(figsize=(5,4))  # (w, h)
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 15
    plt.hist([ind_uncertainties_correct, ind_uncertainties_wrong, ood_uncertainties], 20, 
            density=True, histtype="bar", alpha=1, color=['blue', 'limegreen', 'red'], 
            label=['InC', 'InW', 'OoD'])
    # plt.text(0.45, 11, 'AUROC=%.2f'%(auroc*100), fontsize=fontsize)
    # plt.text(0.45, 9.5, 'AUPR=%.2f'%(aupr*100), fontsize=fontsize)
    # plt.text(0.45, 8, 'FPR95=%.2f'%(fpr95*100), fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xlabel('uncertainty', fontsize=fontsize)
    plt.ylabel('density', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.xlim(0, 0.5e-5)
    # plt.ylim(0, 10.01)
    plt.tight_layout()
    plt.savefig(os.path.join(png_file, 'img_distribution_u.png'))
    plt.savefig(os.path.join(png_file, 'img_distribution_u.pdf'))



if __name__ == "__main__":

    np.random.seed(123)
    args = parse_args()

    main()