import os
import yaml
import numpy as np
import shutil

index_train = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
p = [0.5, 0.1, 0.01, 0.001, 0.0001]
p = [0.08]

for idx in index_train:
    seq_path = '/mnt/data_nas/jcenaa/dataset_av/semantickitti/dataset/sequences/' + idx
    new_seq_path = '/mnt/data_nas/jcenaa/dataset_av/weak_labels_sk_0.08/sequences/' + idx
    print(seq_path, new_seq_path)

    if not os.path.exists(new_seq_path):
        os.makedirs(new_seq_path)
    
    for p_temp in p:
        weak_path = seq_path + '/labels_w_' + str(p_temp)
        weak_path_new = new_seq_path + '/labels_w_' + str(p_temp)
        print(weak_path)
        shutil.copytree(weak_path, weak_path_new)

            