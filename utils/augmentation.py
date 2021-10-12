import torch
import numpy as np

def scale_pointcloud(pointcloud, scale_low=0.75, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, scaled batch of point clouds
    """
    N = pointcloud.shape[0]
    scale_points = pointcloud.clone()
    scales = np.random.uniform(scale_low, scale_high, N)
    for batch_index in range(N):
        scale_points[batch_index, :, :3] *= scales[batch_index]
    return scale_points


def rotate_pointcloud(pointcloud, theta=None, xz=True, yz=False, xy=False):
    N = pointcloud.shape[0]
    rotate_points = pointcloud.clone()
    if theta != None:
        theta = theta
    else:
        theta = np.pi*2 * np.random.uniform()
    rotation_matrix = torch.Tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).cuda()
    for i in range(N):
        if xz:
            rotate_points[i, :, [0, 2]] = rotate_points[i, :, [0, 2]].mm(rotation_matrix)  # random rotation (x,z)
        if yz:
            rotate_points[i, :, [1, 2]] = rotate_points[i, :, [1, 2]].mm(rotation_matrix)  # random rotation (y,z)
        if xy:
            rotate_points[i, :, [0, 1]] = rotate_points[i, :, [0, 1]].mm(rotation_matrix)  # random rotation (x,y)
    return rotate_points


def flip_pointcloud(pointcloud):
    for i in range(pointcloud.shape[0]):
        # switch x, y
        if np.random.rand() < 0.5:
            pointcloud[i, :, [0, 1]] = pointcloud[i, :, [1, 0]]
            pointcloud[i, :, [6, 7]] = pointcloud[i, :, [7, 6]]
        # flip x
        if np.random.rand() < 0.5:
            pointcloud[i, :, 0] = - pointcloud[i, :, 0]
            pointcloud[i, :, 6] = 1 - pointcloud[i, :, 6]
        # flip y
        if np.random.rand() < 0.5:
            pointcloud[i, :, 1] = - pointcloud[i, :, 1]
            pointcloud[i, :, 7] = 1 - pointcloud[i, :, 7]

    return pointcloud