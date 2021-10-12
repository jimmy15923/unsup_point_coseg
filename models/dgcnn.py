import os
import sys
import copy
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)

    device = torch.device('cuda')
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature      # (batch_size, 2*num_dims, num_points, k)

class DGCNN_cls(nn.Module):
    def __init__(self, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.k = 20
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, return_logits=True):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        point_features = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(point_features, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(point_features, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        obj_emb = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(obj_emb)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        if return_logits:
            return x
        else:
            return x, obj_emb, point_features             


class DGCNN_seg(nn.Module):
    def __init__(self):
        super(DGCNN_seg, self).__init__()
        self.k = 20
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Conv1d(256, 40, kernel_size=1, bias=False)

        self.linear1 = nn.Linear(80, 40, bias=False)
        

    def forward(self, x, return_logits=True):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        point_features = self.dp1(x)
        x = self.conv9(point_features)                       # (batch_size, 256, num_points) -> (batch_size, 40, num_points)

        x1 = F.adaptive_max_pool1d(point_features, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(point_features, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        obj_emb = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)       

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)           
        x = self.linear1(x)

        if return_logits:
            return x
        else:
            return x, obj_emb, point_features            




class DGCNN_semseg(nn.Module):
    def __init__(self, args, emb_dims=1024, k=20, is_structure=False, input_channel=9, output_channel=13, with_seg_head=False,
                 with_mutual=False, with_classify=False, with_project_head=False, with_dropout=False, with_linear_seg_head=False, with_attention=False):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        
        self.k = k
        
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.with_project_head = with_project_head
        self.with_seg_head = with_seg_head
        self.with_dropout = with_dropout
        self.with_classify = with_classify
        self.with_linear_seg_head = with_linear_seg_head
        self.with_attention = with_attention
        self.with_mutual = with_mutual
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        if is_structure:
            self.bn7 = nn.BatchNorm1d(768)
            self.bn8 = nn.BatchNorm1d(640)
        else:
            self.bn7 = nn.BatchNorm1d(512)
            self.bn8 = nn.BatchNorm1d(256)
        self.conv1 = nn.Sequential(nn.Conv2d(self.input_channel*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        if is_structure:        
            self.conv7 = nn.Sequential(nn.Conv1d(1216, 768, kernel_size=1, bias=False),
                                       self.bn7,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv8 = nn.Sequential(nn.Conv1d(768, 640, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))
        else:
            self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                       self.bn7,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))        
            
        if self.with_project_head:
            print("Use project head")
            self.project_head = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=True),
                                              nn.ReLU(),
                                              nn.Conv1d(256, 256, kernel_size=1, bias=True))                
        if self.with_seg_head:
            print("Use non-linear head")
            self.bn9 = nn.BatchNorm1d(128)
            self.convseg = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                         self.bn9,
                                         nn.LeakyReLU(negative_slope=0.2),
                                         nn.Conv1d(128, self.output_channel, kernel_size=1, bias=False))
        if self.with_linear_seg_head:
            print("Use linear head")
            self.linearseg = nn.Conv1d(256, self.output_channel, kernel_size=1, bias=False)
            
        if self.with_attention:
            print("Use self attention")
            self.attention = NONLocalBlock1D(256, sub_sample=False, bn_layer=True)
            
        if self.with_mutual:
            print("Use non-local attention")           
            self.mutual_attention = NONLocalBlock1D_mutual(256, sub_sample=False, bn_layer=True)
            
        if self.with_dropout:
            self.point_dp = nn.Dropout(p=0.5)
            
        if self.with_classify:
            self.linear1 = nn.Linear(512, 512, bias=False)
            self.bn9 = nn.BatchNorm1d(512)
            self.dp1 = nn.Dropout(p=0.5)
            self.linear2 = nn.Linear(512, 256)
            self.bn10 = nn.BatchNorm1d(256)
            self.dp2 = nn.Dropout(p=0.5)
            self.linear3 = nn.Linear(256, output_channel)
            self.dp3 = nn.Dropout(p=0.5)
 
    def head_forward(self, point_features, return_gap=False, return_fc1=False):
        batch_size = point_features.size(0)
        num_points = point_features.size(2)        
        x1 = F.adaptive_max_pool1d(point_features, 1).view(batch_size, -1) # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(point_features, 1).view(batch_size, -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        obj_emb = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)
        if return_gap:
            return None, obj_emb, point_features    

        x = F.leaky_relu(self.bn9(self.linear1(obj_emb)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        if return_fc1:
            return None, x, point_features 
        
        x = F.leaky_relu(self.bn10(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        pred = self.linear3(x)   

        return pred, x, point_features            
    

    def forward(self, x, seg_mask=None, return_logits=True, return_gap=False, return_fc1=False):
        inputs = x.clone().permute(0,2,1)
        batch_size = x.size(0)
        num_points = x.size(2)
        if self.input_channel == 9:
            x = get_graph_feature(x, k=self.k, dim9=True)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        else:
            x = get_graph_feature(x, k=self.k)
        if seg_mask != None:
            x = torch.mul(x, seg_mask)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        point_features = self.conv8(x)          # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        
        if self.with_dropout:
            point_features = self.point_dp(point_features)
            
        if self.with_project_head:
            point_features = self.project_head(point_features)  
            
        if self.with_seg_head:   
            if (self.with_attention) or (self.with_mutual):
                point_features = self.attention(point_features)                
            seg_mask = self.convseg(point_features) # (batch_size, 256, num_points) -> (batch_size, 13, num_points)         
            return seg_mask, point_features      

        if self.with_classify:
            x1 = F.adaptive_max_pool1d(point_features, 1).view(batch_size, -1) # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
            x2 = F.adaptive_avg_pool1d(point_features, 1).view(batch_size, -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
            obj_emb = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

            if return_gap:
                return None, obj_emb, point_features                
            
            x = F.leaky_relu(self.bn9(self.linear1(obj_emb)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
            x = self.dp1(x)
            if return_fc1:
                return None, x, point_features            
            
            x = F.leaky_relu(self.bn10(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
            last_x = self.dp2(x)
            x = self.linear3(last_x)                                             # (batch_size, 256) -> (batch_size, output_channels)
           
            if return_logits:
                return x
            else:
                return last_x, obj_emb, point_features               
        
        if self.with_linear_seg_head:
            if (self.with_attention) and (self.with_mutual):
                print("comb")
                self_point_features = self.attention(point_features) 
                mutual_point_features = self.mutual_attention(point_features) 
                point_features = torch.max(torch.cat([self_point_features, mutual_point_features], dim=1), dim=1)[0]
#                 point_features = (self_point_features + _point_features) / 2 
            elif (self.with_attention) and (self.with_mutual == False):
                point_features = self.attention(point_features) 
            elif (self.with_attention==False) and (self.with_mutual):
                point_features = self.mutual_attention(point_features) 
        
            seg_mask = self.linearseg(point_features)
            return seg_mask, point_features  
        
        return inputs, point_features
        