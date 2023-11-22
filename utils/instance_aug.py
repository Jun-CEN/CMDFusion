import os
import yaml
import numpy as np

index_train = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']

for idx in index_train:
    seq_path = '/mnt/data_nas/jcenaa/dataset_av/semantickitti/dataset/sequences/' + idx + '/velodyne'
    ins_path = '/mnt/data_nas/jcenaa/dataset_av/semantickitti/dataset/sequences/' + idx + '/ins'
    ins_labels_path = '/mnt/data_nas/jcenaa/dataset_av/semantickitti/dataset/sequences/' + idx + '/ins_labels'
    if not os.path.exists(ins_path):
        os.makedirs(ins_path)
    if not os.path.exists(ins_labels_path):
        os.makedirs(ins_labels_path)

    g = os.walk(seq_path)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            pts_path = os.path.join(path, file_name)
            print(pts_path)
            label_path = os.path.join(path.replace('velodyne', 'labels'), file_name.replace('bin', 'label'))

            scan = np.fromfile(pts_path, dtype=np.float32)
            scan = scan.reshape((-1, 4))

            label = np.fromfile(label_path, dtype=np.uint32)
            label = label.reshape((-1))

            if label.shape[0] == scan.shape[0]:
                sem_label = label & 0xFFFF  # semantic label in lower half
            else:
                print("Points shape: ", scan.shape)
                print("Label shape: ", sem_label.shape)
                raise ValueError("Scan and Label don't contain same number of points")
            
            CFG = yaml.safe_load(open('/mnt/data_nas/jcenaa/code_av/2DPASS/config/label_mapping/semantic-kitti.yaml', 'r'))
            sem_label_t = np.vectorize(CFG["learning_map"].__getitem__)(sem_label)

            points_ins_keep = []
            label_ins_keep = []
            for i in range(1, 9):
                points_ins_keep.append(scan[sem_label_t==i])
                label_ins_keep.append(sem_label[sem_label_t==i])
            points_ins_keep = np.concatenate(points_ins_keep)[:,:-1]
            label_ins_keep = np.concatenate(label_ins_keep)

            Omega = [0, np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]  # x3
            points_ins_keep_r = []
            label_ins_keep_r = []
            for omega_j in Omega:
                rot_mat = np.array([[np.cos(omega_j),
                                        np.sin(omega_j), 0],
                                    [-np.sin(omega_j),
                                        np.cos(omega_j), 0], [0, 0, 1]])
                points_ins_keep_r.append(np.dot(points_ins_keep, rot_mat))
                label_ins_keep_r.append(label_ins_keep)

            points_ins_keep_r = np.concatenate(points_ins_keep_r)
            label_ins_keep_r = np.concatenate(label_ins_keep_r)

            pts_ins_path = os.path.join(path.replace('velodyne', 'ins'), file_name)
            pts_ins_labels_path = os.path.join(path.replace('velodyne', 'ins_labels'), file_name.replace('bin', 'label'))
            points_ins_keep_r.tofile(pts_ins_path)
            label_ins_keep_r.tofile(pts_ins_labels_path)
            
            pts_n = np.fromfile(pts_ins_path).reshape((-1, 3))
            labels_n = np.fromfile(pts_ins_labels_path, dtype=np.uint32) & 0xFFFF

