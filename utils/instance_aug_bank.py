import os
import yaml
import numpy as np

index_train = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']

for idx in index_train:
    seq_path = '/mnt/data_nas/jcenaa/dataset_av/semantickitti/dataset/sequences/' + idx + '/velodyne'
    ins_path = '/mnt/data_nas/jcenaa/dataset_av/semantickitti/dataset/sequences/' + idx + '/ins_bank/'
    if not os.path.exists(ins_path):
        os.makedirs(ins_path)
    
    for i in range(1, 9):
        ins_path_cls = '/mnt/data_nas/jcenaa/dataset_av/semantickitti/dataset/sequences/' + idx + '/ins_bank/' + str(i)
        if not os.path.exists(ins_path_cls):
            os.makedirs(ins_path_cls)

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

            for i in range(1, 9):
                points_ins_keep = scan[sem_label_t==i][:,:-1]
                # print(points_ins_keep.shape)

                if points_ins_keep.shape[0]:
                    Omega = [0, np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]  # x3
                    points_ins_keep_r = []
                    for omega_j in Omega:
                        rot_mat = np.array([[np.cos(omega_j),
                                                np.sin(omega_j), 0],
                                            [-np.sin(omega_j),
                                                np.cos(omega_j), 0], [0, 0, 1]])
                        points_ins_keep_r.append(np.dot(points_ins_keep, rot_mat))

                    points_ins_keep_r = np.concatenate(points_ins_keep_r)
                    # print(points_ins_keep_r.shape)

                    # pts_ins_path = os.path.join('/mnt/data_nas/jcenaa/dataset_av/semantickitti/dataset/sequences/' + idx + '/ins_bank/' + str(i), file_name)
                    # points_ins_keep_r.tofile(pts_ins_path)
                    # print(pts_ins_path)
                    
                    pts_n = np.fromfile(pts_ins_path).reshape((-1, 3))
                    # print(np.sum(pts_n - points_ins_keep_r))

