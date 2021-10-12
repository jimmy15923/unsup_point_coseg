import os
import argparse
import time
import numpy as np
import scipy.io
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from configs import get_cfg_defaults  
from utils import cal_loss, IOStream, multilabel_acc_score
from sklearn.metrics import jaccard_score
from itertools import combinations
from datetime import datetime
from datasets import ScanObject_coseg, S3DIS_coseg

from sklearn.metrics import precision_score, recall_score, jaccard_score, fbeta_score
from pytorch_metric_learning.losses import NTXentLoss
from utils import cal_loss, IOStream, multilabel_acc_score, farthest_pts_sampling_tensor, scale_pointcloud, rotate_pointcloud, flip_pointcloud
from models import DGCNN_cls, DGCNN_seg, DGCNN_semseg, FCN_sampler, ChamferDistance
from knn_cuda import KNN

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--config', type=str, default="configs/base.yaml")
    parser.add_argument('--obj', type=int, default=4)    
    args = parser.parse_args()        
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config) 
    print(cfg)
    device = torch.device('cuda')

    if cfg.EXP.DATASET == 's3dis':                     
        print("TRAIN on S3DIS")
        cat_to_label = {6:'door', 7:'table', 8:'chair', 9:'sofa', 10:'bookcase'} 
        data_path = cfg.EXP.DATA_PATH
        all_dataset = S3DIS_coseg(area='Area_*', path=data_path, obj=args.obj, num_points=2048, return_color=True)
    else:
        print("TRAIN on ScanObject")
        cat_to_label = {0: 'bag', 1: 'bin', 2: 'box', 3: 'cabinet', 4: 'chair', 5: 'desk',  
                            6: 'display', 7: 'door', 8: 'shelf', 9: 'table', 10: 'bed', 11: 'pillow',
                            12: 'sink', 13: 'sofa', 14: 'toilet', 15: 'all'}        
        all_dataset = ScanObject_coseg(partition='full', permute=args.permute, obj=args.obj, n_points=2048,
                                        raw_path=cfg.EXP.DATA_PATH, return_color=True)

    test_loader = DataLoader(all_dataset, num_workers=2, batch_size=1)

    sampler = FCN_sampler(fg_num_out_points=cfg.MODEL.N_FG, bg_num_out_points=cfg.MODEL.N_BG,
                        group_size=cfg.MODEL.GROUP_SIZE, complete_fps=False, backbone='pointnet', 
                        with_attention=cfg.MODEL.SELF_ATTENTION, with_mutual=cfg.MODEL.MUTUAL_ATTENTION)

    sampler.load_state_dict(torch.load("./work_dirs/raw/{}/{}/model.t7".format(cfg.EXP.NAME, cat_to_label[args.obj])))
    sampler.eval()
    sampler.cuda()

    points = np.zeros(shape=(len(test_loader), cfg.TRAIN.N_POINTS, 3))
    gt_colors = np.zeros(shape=(len(test_loader), cfg.TRAIN.N_POINTS, 3))
    y_preds = np.zeros(shape=(len(test_loader), cfg.TRAIN.N_POINTS))
    y_trues = np.zeros(shape=(len(test_loader), cfg.TRAIN.N_POINTS))

    gt_labels = np.zeros(shape=(len(test_loader), cfg.TRAIN.N_POINTS))

    f1 = []
    # y_trues = []
    with torch.no_grad():
        for i, (coord, feat, label) in enumerate(tqdm(test_loader)):
    #     for i, (coord, label) in enumerate(tqdm(test_loader)):
            # Inference  
            coord = coord.to(device)    
            simp_pc, y_pred = sampler.fg_sampler(coord)
            _, bg_pred = sampler.bg_sampler(coord)
        
            _, fg_idx = KNN(20, transpose_mode=False)(coord.permute(0,2,1).contiguous(), y_pred.permute(0,2,1).contiguous())
            fg_idx = fg_idx.cpu().detach().numpy()
            fg_idx = np.unique(fg_idx)
            y_preds[i][fg_idx] = 1
            
            points[i] = coord.cpu().detach().numpy()
            gt_colors[i] = feat.cpu().detach().numpy()
            y_trues[i] = label.cpu().detach().numpy()        
            f1.append(fbeta_score(y_preds[0], label[0], beta=0.6))
            # y_trues.extend(label[0].cpu().numpy()) 

    scipy.io.savemat('./work_dirs/raw/{}/{}/test_results.mat'.format(cfg.EXP.NAME, cat_to_label[args.obj]),
                     {'coord': points,
                     'color': gt_colors,
                     'y_preds': y_preds,
                     'y_trues': y_trues
                     })     
            
    precision = precision_score(y_trues.flatten(), y_preds.flatten())
    recall = recall_score(y_trues.flatten(), y_preds.flatten())
    f_score = fbeta_score(y_trues.flatten(), y_preds.flatten(), beta=0.75)
    iou = jaccard_score(y_trues.flatten(), y_preds.flatten())
    print("IoU: ", iou)