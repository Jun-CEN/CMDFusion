import os
import argparse
from matplotlib.pyplot import axis
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, roc_curve
from terminaltables import AsciiTable
import torch
import matplotlib.pyplot as plt
from sklearn import manifold
import yaml

# color_map = [[0, 0, 0], [245, 150, 100], [245, 230, 100], [150, 60, 30], [180, 30, 80], [255, 0, 0],
#             [30, 30, 255], [200, 40, 255], [90, 30, 150], [255, 0, 255], ]
CFG = yaml.safe_load(open('/mnt/data_nas/jcenaa/code_av/2DPASS/config/label_mapping/semantic-kitti.yaml', 'r'))
learning_map_inv = np.ones(20)
for i in range(20):
    learning_map_inv[i] = CFG['learning_map_inv'][i]
sem_color_lut = np.zeros((300, 3), dtype=np.float32)
for key, value in CFG['color_map'].items():
    sem_color_lut[key] = value
sem_color_lut /= 255

result_file = '/mnt/data_nas/jcenaa/dataset_av/semantickitti/dataset/sequences/08/bifeatures/000000.npz'
results = np.load(result_file, allow_pickle=True)

img_feat_l = results['img_feat_l']
d2_feat_l = results['d2_feat_l']
d3_feat_l = results['d3_feat_l']
fuse_feat_l = results['fuse_feat_l']
fuse_feat_l_i = results['fuse_feat_l_i']
d2_feat_l_p = results['d2_feat_l_p']
labels = results['labels']
labels_p = results['labels_p']
labels_inv = learning_map_inv[labels]
labels_p_inv = learning_map_inv[labels_p]

print(img_feat_l.shape, d2_feat_l.shape, d3_feat_l.shape, fuse_feat_l.shape, d2_feat_l_p.shape, labels.shape, labels_p.shape)
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
img_feat_list = []
d2_feat_list = []
d3_feat_list = []
fuse_feat_list = []
d2_feat_p_list = []
fuse_feat_i_list = []

for scale in [0,1,2,3]:
    img_feat = img_feat_l[scale]
    d2_feat_p = d2_feat_l_p[scale]
    d2_feat = d2_feat_l[scale][::10]
    d3_feat = d3_feat_l[scale][::10]
    fuse_feat = fuse_feat_l[scale][::10]
    fuse_feat_i = fuse_feat_l_i[scale][::10]

    img_feat_list.append(img_feat)
    d2_feat_list.append(d2_feat)
    d3_feat_list.append(d3_feat)
    fuse_feat_list.append(fuse_feat)
    fuse_feat_i_list.append(fuse_feat_i)
    d2_feat_p_list = [d2_feat_p]

    if 0:
        X = fuse_feat

        X_tsne = tsne.fit_transform(X)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)

        plt.figure(figsize=(5,4))
        point_size = 0.5
        plt.scatter(X_norm[:,0], X_norm[:,1], s=point_size)
        plt.tight_layout()
        plt.savefig('/mnt/data_nas/jcenaa/code_av/2DPASS/ckpt/images/fuse_feat'+ str(scale) + '.png')
        # plt.savefig('/mnt/data_nas/jcenaa/code_av/2DPASS/ckpt/images/img_feat'+ str(scale) + '.pdf')

if 1:
    img_feat_list = np.concatenate(img_feat_list, axis=-1)
    d2_feat_list = np.concatenate(d2_feat_list, axis=-1)
    d3_feat_list = np.concatenate(d3_feat_list, axis=-1)
    fuse_feat_list = np.concatenate(fuse_feat_list, axis=-1)
    fuse_feat_i_list = np.concatenate(fuse_feat_i_list, axis=-1)
    d2_feat_p_list = np.concatenate(d2_feat_p_list, axis=-1)
    print(img_feat_list.shape, d2_feat_list.shape, d3_feat_list.shape, fuse_feat_list.shape, d2_feat_p_list.shape)

    X = img_feat_list
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    plt.figure(figsize=(5,4))
    point_size = 0.5
    labels_tmp = labels_inv[::10].astype(np.int32)
    labels_tmp = labels_p_inv.astype(np.int32)
    print(np.unique(labels_tmp, return_counts=True))
    plt.scatter(X_norm[:,0][labels_tmp != 0], X_norm[:,1][labels_tmp != 0], s=point_size, c=sem_color_lut[labels_tmp[labels_tmp != 0]])
    plt.tight_layout()
    plt.savefig('/mnt/data_nas/jcenaa/code_av/2DPASS/ckpt/images_2/img_feat_list.png')