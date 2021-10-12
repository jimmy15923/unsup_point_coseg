import os
import h5py
import glob
import argparse
import pandas as pd
import numpy as np
import open3d as o3d
from tqdm import tqdm
from collections import Counter

obj_cate = {6:'door', 7:'table', 8:'chair', 9:'sofa', 10:'bookcase'}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--out_dir', type=str, default='/home/jimmy15923/mnt/data/s3dis_coseg/')
    parser.add_argument('--bg_scale', type=int, default=2)
    
    args = parser.parse_args()
    print(args.out_dir)
    df_dict = {}
    fail = []
    for h5file in tqdm(glob.glob("/home/jimmy15923/mnt/data/s3dis/s3dis_h5_07042019/*.h5")):
        file = h5py.File(h5file, 'r+')
        name = os.path.basename(h5file)[:-3]
        raw_points = np.array(file['coords'])
        raw_colors = np.array(file['points'])
        raw_labels = np.array(file['labels'])
        points = raw_points.reshape(-1, 3)[:,0:3]
        colors = raw_colors.reshape(-1, 9)[:,3:6]
        labels = raw_labels.reshape(-1, 2)

        for i in range(labels[:,1].max()): #loop all instances
            obj_class = np.unique(labels[:,0][labels[:,1] == i])[0]
            if obj_class in obj_cate:
#             if label == 8: # get chair
                if not os.path.exists(f"{args.out_dir}/{name}"):
                    os.makedirs(f"{args.out_dir}/{name}")                    

                h5_file_path = f"{args.out_dir}/{name}/{obj_cate[obj_class]}_{i}.h5"
    
                if not os.path.exists(h5_file_path):
                    idx_chair = np.where((labels[:,1] == i) > 0)[0]
                    chair_points = points[idx_chair]
                    chair_colors = colors[idx_chair]
                    chair_labels = labels[idx_chair]

                    chair_pcd = o3d.geometry.PointCloud()
                    chair_pcd.points = o3d.utility.Vector3dVector(chair_points)
                    center_point = chair_pcd.get_center()

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(center_point[None])

                    idx_non_chair = np.where(((labels[:,1]!=i) > 0) & ((labels[:,0]!=obj_class)>0))
                    room_points = points[idx_non_chair]
                    room_colors = colors[idx_non_chair]
                    room_labels = labels[idx_non_chair]                    

                    room_pcd = o3d.geometry.PointCloud()
                    room_pcd.points = o3d.utility.Vector3dVector(room_points)


                    xyz_diff = room_points - center_point
                    distance = np.sum(xyz_diff ** 2, axis=1)
        
              
                    ## By radius
                    x1, x2 = np.min(chair_points[:,0]), np.max(chair_points[:,0])
                    y1, y2 = np.min(chair_points[:,1]), np.max(chair_points[:,1])
                    z1, z2 = np.min(chair_points[:,2]), np.max(chair_points[:,2])
                    w, h, d = x2 - x1, y2 - y1, z2 - z1

#                     long_side = np.mean([w, h, d])
                    long_side *= 1.25
    
                    cond = distance < long_side
                    n_bg_points = np.sum(cond)
            
                    if n_bg_points < 512:
                        continue
                        
                    if n_bg_points > 20480:
                        n_bg_points = 20480
            
                    bg_points = room_points[cond][:n_bg_points]
                    bg_colors = room_colors[cond][:n_bg_points]
                    bg_labels = room_labels[cond][:n_bg_points]
                    

                    print(obj_cate[obj_class], n_bg_points)

                    object_with_bg_points = np.concatenate([bg_points, chair_points])
                    object_with_bg_colors = np.concatenate([bg_colors, chair_colors])
                    object_with_bg_labels = np.concatenate([bg_labels, chair_labels])
                    
                    xyz_min = np.amin(object_with_bg_points, axis=0)[0:3]
                    object_with_bg_points[:, 0:3] -= xyz_min
                    
                    fp = h5py.File(f"{args.out_dir}/{name}/{obj_cate[obj_class]}_{i}.h5", 'w')

                    fp.create_dataset('data', data=object_with_bg_points[:,[0,2,1]], compression='gzip', dtype='float32')
                    fp.create_dataset('color', data=object_with_bg_colors, compression='gzip', dtype='float32') 
                    fp.create_dataset('label', data=object_with_bg_labels, compression='gzip', dtype='int32')
                    fp.close()


