# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'ops', 'pt_custom_ops'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from modules import PointsObjClsModule

class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0, width=1, depth=2, score_sampling=False):
        super().__init__()
        self.depth = depth
        self.width = width

        self.sa1 = PointnetSAModuleVotes(
            npoint=2048,
            radius=0.2,
            nsample=64,
            mlp=[input_feature_dim] + [64 * width for i in range(depth)] + [128 * width],
            use_xyz=True,
            normalize_xyz=True
        )
        self.ss1 = PointsObjClsModule(128 * width + 3) if score_sampling else None

        self.sa2 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.4,
            nsample=32,
            mlp=[128 * width] + [128 * width for i in range(depth)] + [256 * width],
            use_xyz=True,
            normalize_xyz=True
        )
        self.ss2 = PointsObjClsModule(256 * width + 3) if score_sampling else None

        self.sa3 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.8,
            nsample=16,
            mlp=[256 * width] + [128 * width for i in range(depth)] + [256 * width],
            use_xyz=True,
            normalize_xyz=True
        )
        self.ss3 = PointsObjClsModule(256 * width + 3) if score_sampling else None

        self.sa4 = PointnetSAModuleVotes(
            npoint=256,
            radius=1.2,
            nsample=16,
            mlp=[256 * width] + [128 * width for i in range(depth)] + [256 * width],
            use_xyz=True,
            normalize_xyz=True
        )

        self.fp1 = PointnetFPModule(mlp=[256 * width + 256 * width, 256 * width, 256 * width])
        self.fp2 = PointnetFPModule(mlp=[256 * width + 256 * width, 256 * width, 288])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        if self.ss1 is not None:
            end_points['ss1_scores'] = self.ss1(torch.cat([xyz.permute(0,2,1), features], axis=1))
            ss1_scores = end_points['ss1_scores'].detach().squeeze(1)
            inds = torch.topk(ss1_scores, 1024)[1].int()
        else:
            inds = None

        xyz, features, fps_inds = self.sa2(xyz, features, inds)  # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        if self.ss2 is not None:
            end_points['ss2_scores'] = self.ss2(torch.cat([xyz.permute(0,2,1), features], axis=1))
            ss2_scores = end_points['ss2_scores'].detach().squeeze(1)
            inds = torch.topk(ss2_scores, 512)[1].int()
        else:
            inds = None

        xyz, features, fps_inds = self.sa3(xyz, features, inds)  # this fps_inds is just 0,1,...,511
        end_points['sa3_inds'] = fps_inds
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        if self.ss3 is not None:
            end_points['ss3_scores'] = self.ss3(torch.cat([xyz.permute(0,2,1), features], axis=1))
            ss3_scores = end_points['ss3_scores'].detach().squeeze(1)
            inds = torch.topk(ss3_scores, 256)[1].int()
        else:
            inds = None

        xyz, features, fps_inds = self.sa4(xyz, features, inds)  # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'],
                            end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        # print(end_points['sa1_inds'].shape)
        end_points['fp2_inds'] = end_points['sa1_inds'][:, 0:num_seed]  # indices among the entire input point clouds

        return end_points


if __name__ == '__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16, 20000, 6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
