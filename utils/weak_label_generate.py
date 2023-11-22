import os
import yaml
import numpy as np

index_train = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
p = 0.08

for idx in index_train:
    seq_path = '/mnt/data_nas/jcenaa/dataset_av/semantickitti/dataset/sequences/' + idx + '/labels'
    w_seq_path = '/mnt/data_nas/jcenaa/dataset_av/semantickitti/dataset/sequences/' + idx + '/labels_w_' + str(p)

    if not os.path.exists(w_seq_path):
        os.makedirs(w_seq_path)
    g = os.walk(seq_path)
    for path, dir_list, file_list in g:
        print(path)
        for file_name in file_list:
            label_path = os.path.join(path, file_name)
            w_label_path = os.path.join(w_seq_path, file_name)
            print(w_label_path)
            annotated_data = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            mask = np.zeros_like(annotated_data)
            mask[:int(mask.shape[0] * p)] = 1
            mask = 1 - mask
            np.random.shuffle(mask)
            mask = mask.astype('bool')
            annotated_data[mask.squeeze()] = 0
            annotated_data.tofile(w_label_path)