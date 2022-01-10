import os
import h5py
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from .pcd import normalize_data, center_data, translate_pointcloud, ori_jitter_pointcloud, ori_rotate_pointcloud


def load_data_cls(partition, data_dir):
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40*hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

class ModelNet40(Dataset):
    def __init__(self, num_points, data_dir='/home/jimmy15923/mnt/data/', partition='train'):
        self.data, self.label = load_data_cls(partition, data_dir)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item]

        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = ori_jitter_pointcloud(pointcloud)
            pointcloud = ori_rotate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
            pointcloud = pointcloud[:self.num_points]

        pointcloud = center_data(pointcloud)
        pointcloud = normalize_data(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]