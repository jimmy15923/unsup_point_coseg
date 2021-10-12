import torch
import torch.nn as nn
from .src import SampleNet

class FCN_sampler(nn.Module):
    def __init__(self, fg_num_out_points=512, bg_num_out_points=512, complete_fps=False, group_size=7, backbone='pointnet',
                 skip_projection=False):
        super(FCN_sampler, self).__init__() 
        self.fg_sampler = SampleNet(
                    num_out_points=fg_num_out_points,
                    bottleneck_size=256,
                    group_size=group_size,
                    backbone=backbone,
                    initial_temperature=1.0,
                    input_shape="bnc",
                    output_shape="bnc",
                    complete_fps=complete_fps,
                    skip_projection=skip_projection)  

        self.bg_sampler = SampleNet(
                    num_out_points=bg_num_out_points,
                    bottleneck_size=256,
                    group_size=group_size,
                    backbone=backbone,
                    initial_temperature=1.0,
                    input_shape="bnc",
                    output_shape="bnc",
                    complete_fps=complete_fps,
                    skip_projection=skip_projection) 

    def forward(self, x, fg=True):
        if fg == True:
            simp_pc, proj_pc = self.fg_sampler(x)
            return simp_pc, proj_pc            
        else:
            simp_pc, proj_pc = self.bg_sampler(x)              
            return simp_pc, proj_pc