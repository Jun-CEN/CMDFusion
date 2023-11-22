import os
import yaml
import numpy as np

index_train = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
p = 0.08

for idx in index_train:
    seq_path = '/mnt/data_nas/jcenaa/dataset_av/semantickitti/dataset/sequences/' + idx + '/labels_w_' + str(p)

    g = os.walk(seq_path)
    for path, dir_list, file_list in g:
        print(path)
        for file_name in file_list:
            label_path = os.path.join(path, file_name)
            annotated_data = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            print(np.sum(annotated_data != 0)/annotated_data.shape[0])
            