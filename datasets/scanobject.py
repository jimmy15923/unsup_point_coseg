import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from .pcd import normalize_data, center_data, ori_rotate_pointcloud

class ScanObject_coseg(Dataset):
    """Scan object dataset
    permute: ['augmented25rot', 'augmented25_norot', 'augmentedrot', 'augmentedrot_scale75', 'raw']
    obj: {0: 'bag', 1: 'bin', 2: 'box', 3: 'cabinet', 4: 'chair', 5: 'desk',
          6: 'display', 7: 'door', 8: 'shelf', 9: 'table', 10: 'bed', 11: 'pillow',
          12: 'sink', 13: 'sofa', 14: 'toilet', 15: 'all'}
    """
    def __init__(self, raw_path='/home/jimmy15923/docker_mnt/data/scanobjectnn/h5_files/main_split',
                n_points=1024, partition='training', permute='raw', obj=4, label_binarize=True, norm=True, center=True,
                mask=True):       
        cat_to_label = {0: 'bag', 1: 'bin', 2: 'box', 3: 'cabinet', 4: 'chair', 5: 'desk',
                        6: 'display', 7: 'door', 8: 'shelf', 9: 'table', 10: 'bed', 11: 'pillow',
                        12: 'sink', 13: 'sofa', 14: 'toilet', 15: 'all'}

        print(f"Select {cat_to_label[obj]} object")
        self.partition = partition
        self.n_points = n_points
        self.norm = norm
        self.center = center
        if permute == 'raw':
            self.data = h5py.File("{}/{}_objectdataset.h5".format(raw_path, partition), 'r+')
        else:
            self.data = h5py.File("{}/{}_objectdataset_{}.h5".format(raw_path, partition, permute), 'r+')    
            
        self.points = np.array(self.data['data'])
        self.labels = np.array(self.data['label'])

        self.masks = np.array(self.data['mask'])
        self.data.close()
        
        select_label = cat_to_label[obj]

        if select_label != 'all':
            self.points = self.points[self.labels == obj]
            self.masks = self.masks[self.labels == obj]
        if label_binarize:
            self.masks[self.masks != -1] = 1
            self.masks[self.masks == -1] = 0  
        else:
            self.masks += 1 # 0 stands for BG
            
        print("Number of data: ", self.points.shape[0])
        
    def __getitem__(self, item):
        coord = self.points[item]
        label = self.masks[item]

        if self.partition == "training":
            coord = ori_rotate_pointcloud(coord)
            indices = list(range(coord.shape[0]))
            np.random.shuffle(indices)
            coord = coord[indices[:self.n_points]]
            label = label[indices[:self.n_points]]

        if self.center:
            coord = center_data(coord)
        if self.norm:
            coord = normalize_data(coord)

        return coord, label.astype(np.int64)

    def __len__(self):
        return len(self.points)