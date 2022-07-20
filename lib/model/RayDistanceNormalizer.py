import torch
import torch.nn as nn
import torch.nn.functional as F

class RayDistanceNormalizer(nn.Module):
    def __init__(self, opt):
        super(RayDistanceNormalizer, self).__init__()

        self.opt = opt
        self.norm_method = self.opt.rdist_norm

        if self.norm_method == 'uvf':
            self.half_w = (self.opt.img_size[0] // 2)
            self.half_h = (self.opt.img_size[1] // 2)
        if self.norm_method == 'minmax':
            CAM_Bz_SHIFT = self.opt.wks_z_shift
            Bx_SIZE = self.opt.wks_size[0] / 2
            By_SIZE = self.opt.wks_size[1] / 2
            Bz_SIZE = self.opt.wks_size[2] / 2
            self.rdist_min = -Bz_SIZE + CAM_Bz_SHIFT
            self.rdist_max = torch.norm(torch.tensor([Bx_SIZE, By_SIZE, Bz_SIZE + CAM_Bz_SHIFT], dtype=torch.float)).item()

    def forward(self, queries, norm_uv=None, transforms=None, calibs=None):
        '''
        Normalize dist_ray_feature
        :param dist_ray_feature: [B, 1, N] query distance along the ray normalized by projected uv distance along the ray
        :return:
        '''
        batch_size = queries.shape[0]
        pt_size = queries.shape[2]
        # (B, 1, N) = (B, 3, N)
        abs_dist_ray = torch.norm(queries, dim=1).unsqueeze(1)

        if self.norm_method == 'uvf':
            # (B, 2, 3)
            inv_trans = torch.zeros_like(transforms)    
            inv_trans[:, 0,0] = self.half_w
            inv_trans[:, 1,1] = self.half_h
            # inv_trans[:, 0,2] = self.half_w 
            # inv_trans[:, 1,2] = self.half_h
            inv_trans[:, 0,2] = 0
            inv_trans[:, 1,2] = 0
            scale = inv_trans[:, :2, :2]
            shift = inv_trans[:, :2, 2:3]
            # (B, 2, N)
            uv = torch.baddbmm(shift, scale, norm_uv)
            # (B)
            ave_focal = (calibs[:, 0,0] + calibs[:, 1,1]) / 2
            # (B, 1, N)
            ave_focal = ave_focal.unsqueeze(1).expand(batch_size, pt_size).unsqueeze(1)
            # (B, 3, N)
            proj_uvf = torch.cat((uv, ave_focal), dim=1)
            # (B, 1, N)
            proj_dist_ray = torch.norm(proj_uvf, dim=1).unsqueeze(1)
            
            return abs_dist_ray / proj_dist_ray

        elif self.norm_method == 'minmax':
            
            return (abs_dist_ray - self.rdist_min) / (self.rdist_max - self.rdist_min)
            
