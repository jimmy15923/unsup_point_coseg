import os
import h5py
import glob
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from .pcd import normalize_data, center_data, ori_rotate_pointcloud
try:
    from pointnet2.utils.pointnet2_utils import furthest_point_sample 
except:
    pass

def farthest_pts_sampling_tensor(pts, num_samples, return_sampled_idx=False):
    '''

    :param pts: bn, n, 3
    :param num_samples:
    :return:
    '''
    sampled_pts_idx = furthest_point_sample(pts, num_samples)
    sampled_pts_idx_viewed = sampled_pts_idx.view(sampled_pts_idx.shape[0]*sampled_pts_idx.shape[1]).cuda().type(torch.LongTensor)
    batch_idxs = torch.tensor(range(pts.shape[0])).type(torch.LongTensor)
    batch_idxs_viewed = batch_idxs[:, None].repeat(1, sampled_pts_idx.shape[1]).view(batch_idxs.shape[0]*sampled_pts_idx.shape[1])
    sampled_pts = pts[batch_idxs_viewed, sampled_pts_idx_viewed, :]
    sampled_pts = sampled_pts.view(pts.shape[0], num_samples, 3)

    if return_sampled_idx == False:
        return sampled_pts
    else:
        return sampled_pts, sampled_pts_idx


def load_data_coseg(path="/home/jimmy15923/mnt/data/s3dis_instance_bg_points/", area="Area_*", obj='chair', num_points=2048, pre_process=True):
    all_files = glob.glob(f"{path}/{area}/{obj}*.h5")

    coord_list, color_list, label_list = [], [], []
    for f in tqdm(all_files):
        file = h5py.File(f, 'r+')
        coord = file["data"][:]
        color = file["color"][:]
        label = file["label"][:]
        file.close()
        
        # if coord.shape[0] < num_points:
        #     continue            
            
#         if num_points > num_points:
        _, indices = farthest_pts_sampling_tensor(torch.tensor(coord)[None].cuda(), num_points, return_sampled_idx=True)
        indices = indices.cpu().numpy()[0]
        coord = coord[indices]
        color = color[indices]
        label = label[indices,0]
        
#         coord = coord[:num_points,:]
#         color = color[:num_points]
#         label = label[:num_points,0]       

        coord_list.append(coord)
        color_list.append(color)
        label_list.append(label)

    coords = np.array(coord_list)
    colors = np.array(color_list)
    labels = np.array(label_list, dtype=np.int32)

    return coords, colors, labels



class S3DIS_coseg(Dataset):
    def __init__(self, path="/home/jimmy15923/mnt/data/s3dis_instance_bg_points", partition='train', area="Area_*", obj=6,
                 num_points=10240, label_binarize=True, norm=True, center=True, return_color=False):
        
        cat_to_label = {6:'door', 7:'table', 8:'chair', 9:'sofa', 10:'bookcase'}
        cat = cat_to_label[obj]
        print(f"Select {cat} object")
        data = load_data_coseg(path=path, area=area, obj=cat, num_points=num_points, pre_process=True)
        self.coord = data[0]
        self.feat = data[1]
        self.label = data[2]

        self.return_color = return_color
        if label_binarize:
            self.label[self.label != obj] = 0
            self.label[self.label == obj] = 1      
                      
        self.partition = partition
        self.norm = norm
        self.center = center
        self.label_binarize = label_binarize
        
    def __getitem__(self, item):
        coord = self.coord[item]
        feat = self.feat[item]
        label = self.label[item]
        
        if self.center:
            coord = center_data(coord)
        if self.norm:
            coord = normalize_data(coord)        
        
        if self.partition == 'train':
            indices = list(range(coord.shape[0]))
            np.random.shuffle(indices)
            coord = coord[indices]
            feat = feat[indices]
            label = label[indices]

        if self.return_color:
            return coord, feat, label           
        else:
            return coord, label

    def __len__(self):
        return len(self.coord)
   
    