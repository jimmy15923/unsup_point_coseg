from __future__ import print_function
import os
import argparse
import time
import numpy as np
import scipy.io
import logging
from itertools import combinations
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score, recall_score, jaccard_score, fbeta_score
from pytorch_metric_learning.losses import NTXentLoss

from configs import get_cfg_defaults  
from datetime import datetime
from datasets import ScanObject_coseg, S3DIS_coseg
from utils import cal_loss, IOStream, multilabel_acc_score, farthest_pts_sampling_tensor, scale_pointcloud, rotate_pointcloud, flip_pointcloud
from models import DGCNN_cls, DGCNN_seg, DGCNN_semseg, FCN_sampler, ChamferDistance
from knn_cuda import KNN

torch.manual_seed(7)
torch.backends.cudnn.deterministic = True
np.random.seed(7)

def train(args, io):
    if cfg.EXP.DATASET == 's3dis':                     
        print("TRAIN on S3DIS")
        data_path = cfg.EXP.DATA_PATH
        train_dataset = S3DIS_coseg(area='Area_*', path=data_path, obj=args.obj, num_points=cfg.TRAIN.N_POINTS)
        test_dataset = S3DIS_coseg(area='Area_5_*', path=data_path, obj=args.obj, num_points=cfg.TRAIN.N_POINTS)
    else:
        print("TRAIN on ScanObject")
        train_dataset = ScanObject_coseg(partition='full', permute=args.permute, obj=args.obj, n_points=cfg.TRAIN.N_POINTS,
                                        raw_path=cfg.EXP.DATA_PATH, return_color=False)
        test_dataset = ScanObject_coseg(partition='test', permute=args.permute, obj=args.obj, n_points=cfg.TRAIN.N_POINTS,
                                        raw_path=cfg.EXP.DATA_PATH, return_color=False)
    cfg.TRAIN.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE if len(train_dataset) > cfg.TRAIN.BATCH_SIZE else len(train_dataset)
         
    train_loader = DataLoader(train_dataset, num_workers=2, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=2, batch_size=1)

    device = torch.device("cuda:0")

    sampler = FCN_sampler(fg_num_out_points=cfg.MODEL.N_FG,
                          bg_num_out_points=cfg.MODEL.N_BG,
                          group_size=cfg.MODEL.GROUP_SIZE)
    sampler.to(device)

    feature_extractor = DGCNN_cls()
    feature_extractor = nn.DataParallel(feature_extractor)
    feature_extractor.to(device)  
    feature_extractor.load_state_dict(torch.load(cfg.EXP.PRETRAIN_MODEL_PATH))          
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()        
  
    logging.info("Let's use", torch.cuda.device_count(), "GPUs!")

    if cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
        print("Use SGD")
        opt = optim.SGD(sampler.parameters(), lr=cfg.TRAIN.LR*10, momentum=0.9, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.AdamW(sampler.parameters() , lr=cfg.TRAIN.LR, weight_decay=1e-4)

    if cfg.TRAIN.SCHEDULER == 'cos':
        scheduler = CosineAnnealingLR(opt, cfg.TRAIN.N_EPOCHS, eta_min=1e-3)
    elif cfg.TRAIN.SCHEDULER == 'step':
        scheduler = StepLR(opt, 20, 0.5, cfg.TRAIN.N_EPOCHS)
 
    point_nxt_loss = NTXentLoss(temperature=0.5)
    obj_nxt_loss = NTXentLoss(temperature=cfg.TRAIN.TEMP)
    cs_sim = nn.CosineSimilarity(dim=1)

    ####################
    #TRAIN  
    ####################
    best_iou = 0
    best_f1 = 0
    point_loss = torch.zeros(1).cuda()
    obj_loss = torch.zeros(1).cuda()
    repulsion_loss = torch.zeros(1).cuda()  
    for epoch in range(cfg.TRAIN.N_EPOCHS):
        sampler.requires_grad_(True)
        sampler.train()

        running_loss = 0
        running_obj_loss = 0
        running_point_loss = 0
        running_sample_loss = 0
        running_rep = 0

        for i, (coord, label) in enumerate(train_loader):
            coord, label = coord.to(device), label.to(device)                  
            coord = farthest_pts_sampling_tensor(coord, cfg.TRAIN.N_POINTS)

            fg_simp_pc, fg_coord = sampler.fg_sampler(coord)
            bg_simp_pc, bg_coord = sampler.bg_sampler(coord)

            ## REPUL_LOSS            
            same_point_loss1, same_point_loss2 = ChamferDistance()(fg_coord, bg_coord)
            repulsion_loss1 = torch.mean(torch.max(torch.tensor(cfg.MODEL.REPULSION).cuda() - \
                same_point_loss1, torch.tensor(0.0).cuda())[0])
            repulsion_loss2 = torch.mean(torch.max(torch.tensor(cfg.MODEL.REPULSION).cuda() - \
                same_point_loss2, torch.tensor(0.0).cuda())[0])
            repulsion_loss = (repulsion_loss1 + repulsion_loss2)
                           

            obj_logits, object_features, obj_point_features = feature_extractor(fg_coord.permute(0,2,1).contiguous(),
                                                                                return_logits=False) 
            bg_logits, background_features, bg_point_features = feature_extractor(bg_coord.permute(0,2,1).contiguous(), 
                                                                                  return_logits=False)       

            # SampleNet losses
            fg_simplification_loss = sampler.fg_sampler.get_simplification_loss(coord, fg_simp_pc,
                                                                            cfg.MODEL.N_FG,
                                                                            gamma=cfg.MODEL.FG_GAMMA)
            fg_projection_loss = sampler.fg_sampler.get_projection_loss()

            fg_loss = cfg.MODEL.ALPHA * fg_simplification_loss + cfg.MODEL.LAMDA * fg_projection_loss 

            bg_simplification_loss = sampler.bg_sampler.get_simplification_loss(coord, bg_simp_pc,
                                                                            cfg.MODEL.N_BG,
                                                                            gamma=cfg.MODEL.BG_GAMMA)
            bg_projection_loss = sampler.bg_sampler.get_projection_loss()
            bg_loss = cfg.MODEL.ALPHA * bg_simplification_loss + cfg.MODEL.LAMDA * bg_projection_loss 

            samplenet_loss = fg_loss + bg_loss 

            # POINT_LOSS:
            point_loss = []
            for i in range(cfg.TRAIN.BATCH_SIZE):               
                object_feature = obj_point_features[i].permute(1,0)
                background_feature = bg_point_features[i].permute(1,0)
                embeddings = torch.cat([object_feature, background_feature], dim=0)

                point_labels = torch.cat([torch.zeros(object_feature.shape[0]).long(),
                                            torch.arange(1, embeddings.shape[0]-object_feature.shape[0]+1)]).long().cuda()

                ### create N_POS_PAIRS positive pairs for effcient GPU###        
                positives = torch.randint(low=0, high=object_feature.shape[0], size=(cfg.TRAIN.N_POS_PAIRS, 2)).to(point_labels.device)

                ### create N_NEG_PAIRS negative pairs ###
                negatives = torch.randint(low=0, high=background_feature.shape[0], size=(cfg.TRAIN.N_NEG_PAIRS, 2)).to(point_labels.device)
                negatives[:,1] += object_feature.shape[0] 
                a1, p = positives[:,0], positives[:,1]
                a2, n = negatives[:,0], negatives[:,1]
                indices_tuple = (a1, p, a2, n)

                point_loss_batch = point_nxt_loss(embeddings, point_labels, indices_tuple)                    

                point_loss.append(point_loss_batch)
            point_loss = torch.mean(torch.stack(point_loss))
                
            ## OBJ_LOSS:
            embeddings = torch.cat([object_features, background_features], dim=0)
            
            obj_labels = torch.cat([torch.zeros(object_features.shape[0]).long(),
                                torch.arange(1, background_features.shape[0]+1)]).long().cuda()

            obj_loss = obj_nxt_loss(embeddings, obj_labels)   

            loss = point_loss + obj_loss + samplenet_loss + repulsion_loss 

            opt.zero_grad()           
            loss.backward()
            opt.step()

            with torch.no_grad():
                running_loss += loss.item()


        # Validation
        y_preds = np.zeros(shape=(len(test_loader), cfg.TRAIN.N_POINTS))
        bg_preds = np.zeros(shape=(len(test_loader), cfg.TRAIN.N_POINTS))
        points = np.zeros(shape=(len(test_loader), cfg.TRAIN.N_POINTS, 3))
        y_trues = []
        f1 = []
        sampler.eval()
        sampler.requires_grad_(False)
        n_spread = 20

        with torch.no_grad():
            for i, (coord, label) in enumerate(test_loader):
                # Inference  
                coord = coord.to(device)   
                coord, fps_idx = farthest_pts_sampling_tensor(coord, cfg.TRAIN.N_POINTS, return_sampled_idx=True)

                simp_pc, y_pred = sampler.fg_sampler(coord)
                _, bg_pred = sampler.bg_sampler(coord)             

                _, idx = KNN(n_spread, transpose_mode=False)(coord.permute(0,2,1).contiguous(), y_pred.permute(0,2,1).contiguous())
                idx = idx.cpu().detach().numpy()
                best_idx = np.unique(idx)     

                y_preds[i][best_idx] = 1
            
                points[i] = coord.cpu().detach().numpy()
                y_trues.extend(label[0][fps_idx[0].long()].cpu().numpy())   

        precision = precision_score(y_trues, y_preds.flatten())
        recall = recall_score(y_trues, y_preds.flatten())
        f_score = fbeta_score(y_trues, y_preds.flatten(), beta=0.75)
        iou = jaccard_score(y_trues, y_preds.flatten())    

        io.cprint('{}, Epoch {}, loss/obj/point {:.3f}, iou {:.3f}, pre/rec: {:.3f}/{:.3f}/{:.3f}'.format(
            str(datetime.now())[:19], epoch, running_loss, iou, precision, recall, best_iou))  
        
        if (f_score > best_f1) & (epoch > 50):
            best_iou = iou
            best_f1 = f_score
            io.cprint("Best IOU/F1/Pre = {}/{}/{}".format(best_iou, best_f1, precision))
            torch.save(sampler.state_dict(), 'work_dirs/{}/{}/{}/model.t7'.format(args.permute, cfg.EXP.NAME, cat_to_label[args.obj]))      
               
        torch.save(sampler.state_dict(), 'work_dirs/{}/{}/{}/model_last.t7'.format(args.permute, cfg.EXP.NAME, cat_to_label[args.obj]))    
           
        if cfg.TRAIN.SCHEDULER == 'cos':
            scheduler.step()
        elif cfg.TRAIN.SCHEDULER == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5 

    # Run testing 
    sampler.load_state_dict(torch.load('work_dirs/{}/{}/{}/model.t7'.format(args.permute, cfg.EXP.NAME, cat_to_label[args.obj])))
    
    os.rename('work_dirs/{}/{}/{}/run.log'.format(args.permute, cfg.EXP.NAME, cat_to_label[args.obj]),
              'work_dirs/{}/{}/{}/{}_run_iou{:.2f}_f{:.2f}.log'.format(args.permute, cfg.EXP.NAME, 
               cat_to_label[args.obj], str(datetime.now())[:19], best_iou*100, best_f1*100))
               

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--config', type=str, default="configs/base.yaml")
    parser.add_argument('--obj', type=int, default=4)
    parser.add_argument('--permute', type=str, default="raw",
    help=['augmented25rot', 'augmented25_norot', 'augmentedrot', 'augmentedrot_scale75', 'raw', 's3dis'])

    if __name__ == "__main__":
        args = parser.parse_args()        
        cfg = get_cfg_defaults()
        cfg.merge_from_file(args.config) 
        print(cfg)
        if cfg.EXP.DATASET == "s3dis":
            cat_to_label = {6:'door', 7:'table', 8:'chair', 9:'sofa', 10:'bookcase'}            
        else:         
            cat_to_label = {0: 'bag', 1: 'bin', 2: 'box', 3: 'cabinet', 4: 'chair', 5: 'desk',
                            6: 'display', 7: 'door', 8: 'shelf', 9: 'table', 10: 'bed', 11: 'pillow',
                            12: 'sink', 13: 'sofa', 14: 'toilet', 15: 'all'}
        print("Training on", cat_to_label[args.obj], "with permute: ", args.permute)
        if not os.path.exists('work_dirs/{}/{}/{}'.format(args.permute, cfg.EXP.NAME, cat_to_label[args.obj])):
            os.makedirs('work_dirs/{}/{}/{}'.format(args.permute, cfg.EXP.NAME, cat_to_label[args.obj]))

        # Save training config
        with open('work_dirs/{}/{}/{}/config.yaml'.format(args.permute, cfg.EXP.NAME, cat_to_label[args.obj]), "w") as f:
            f.write(cfg.dump()) 
        
        io = IOStream('work_dirs/{}/{}/{}/run.log'.format(args.permute, cfg.EXP.NAME, cat_to_label[args.obj]))
        io.cprint(str(args))
        train(args, io)
