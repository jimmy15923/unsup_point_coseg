import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

def normalize_data(pc):
    # for pc in pcs:
    #get furthest point distance then normalize
    d = max(np.sum(np.abs(pc)**2, axis=-1)**(1./2))
    pc /= d

    # pc[:,0]/=max(abs(pc[:,0]))
    # pc[:,1]/=max(abs(pc[:,1]))
    # pc[:,2]/=max(abs(pc[:,2]))

    return pc

#USE For SUNCG, to center to origin
def center_data(pc):
    # for pc in pcs:
    centroid = np.mean(pc, axis=0)
    pc[:,0] -= centroid[0]
    pc[:,1] -= centroid[1]
    pc[:,2] -= centroid[2]

    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def ori_jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def ori_rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud