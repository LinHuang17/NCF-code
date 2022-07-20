import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier
from .RayDistanceNormalizer import RayDistanceNormalizer
from .HGFilters import *
from ..net_util import init_net


class HGPIFuNet(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 opt,
                 projection_mode='perspective',
                 sdf_loss_term=nn.L1Loss(),
                 xyz_loss_term=nn.SmoothL1Loss(),
                 ):
        super(HGPIFuNet, self).__init__(
            projection_mode=projection_mode,
            sdf_loss_term=sdf_loss_term,
            xyz_loss_term=xyz_loss_term)

        self.name = 'hgpifu'

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter = HGFilter(opt)

        self.last_op = None
        if self.opt.out_type[-3:] == 'sdf':
            if self.opt.use_tanh:
                self.last_op = nn.Tanh()
        if self.opt.use_xyz:
            mlp_dim = self.opt.mlp_dim_xyz
        else:
            mlp_dim = self.opt.mlp_dim

        self.surface_classifier = SurfaceClassifier(
                                                    filter_channels=mlp_dim,
                                                    num_views=self.opt.num_views,
                                                    no_residual=self.opt.no_residual,
                                                    last_op=self.last_op)

        self.normalizer = RayDistanceNormalizer(opt)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []

        # init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        # gain (float)       -- scaling factor for normal, xavier and orthogonal.
        # init_net(self)
        init_net(self, init_type=self.opt.init_type, init_gain=self.opt.init_gain)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, points, calibs, transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        if labels is not None:
            self.labels = labels

        self.uvz = self.projection(points, calibs, transforms)
        uv = self.uvz[:, :2, :]
        z = self.uvz[:, 2:3, :]

        # debug for query during forward
        # pdb.set_trace()
        # debug_dir = '/mnt/data0/lin/results/hopifu/debug/forward_query/'
        # res = {'img': images.cpu()[0], 'samples': xyz.cpu()[0], 'labels': labels.cpu()[0]}
        # viz_debug_query_forward(res, 1, debug_dir)

        in_img = (uv[:, 0] >= -1.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= -1.0) & (uv[:, 1] <= 1.0)

        # self.z_feat = self.normalizer(z, calibs=calibs)
        self.dist_ray_feat = self.normalizer(points, uv, transforms=transforms, calibs=calibs)

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, uv)

        self.intermediate_preds_list = []

        for im_feat in self.im_feat_list:
            # [B, Feat_i + z, N]
            point_local_feat_list = [self.index(im_feat, uv), self.dist_ray_feat]

            if self.opt.skip_hourglass:
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)

            # out of image plane is always set to 0 for occupancy or 1000 for sdf
            # pred (B, 1, 5000)/(B, 4, 5000)
            # in_img (B, N), not_in_img (B, 1, N)
            # ((in_img == False).nonzero(as_tuple=True))
            pred = in_img[:,None].float() * self.surface_classifier(point_local_feat)
            if self.opt.out_type[-3:] == 'sdf':
                norm_factor = (self.opt.clamp_dist / self.opt.norm_clamp_dist)
                not_in_img = (torch.logical_not(in_img).float() * (100 * self.opt.clamp_dist / norm_factor)).unsqueeze(1)
                if self.opt.use_xyz:
                    added_zeros = torch.zeros((pred.shape[0], 3, pred.shape[2])).cuda()
                    pred = pred + torch.cat((not_in_img, added_zeros), dim=1)
                else:
                    pred = pred + not_in_img
            self.intermediate_preds_list.append(pred)

        # shape (B, 1, 5000)
        if self.opt.use_xyz:
            self.preds = self.intermediate_preds_list[-1][:,0,:].unsqueeze(1)
            # shape (B, 3, 5000)
            self.xyzs = self.intermediate_preds_list[-1][:,1:,:]
        else:
            self.preds = self.intermediate_preds_list[-1]

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    # def get_error(self):
    #     '''
    #     Hourglass has its own intermediate supervision scheme
    #     '''
    #     error = 0
    #     for preds in self.intermediate_preds_list:
    #         error += self.error_term(preds, self.labels)
    #     error /= len(self.intermediate_preds_list)

    #     return error
    def get_loss(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        loss_dict = {}
        loss_dict['sdf_loss'] = 0.
        if self.opt.use_xyz:
            loss_dict['xyz_loss'] = 0.
        loss_dict['total_loss'] = 0.
        for preds in self.intermediate_preds_list:

            if self.opt.out_type[-3:] == 'sdf':
                pred_sdf = torch.clamp(preds[:,0,:].unsqueeze(1), -self.opt.norm_clamp_dist, self.opt.norm_clamp_dist)
                gt_sdf = torch.clamp(self.labels, -self.opt.norm_clamp_dist, self.opt.norm_clamp_dist)
                loss_dict['sdf_loss']  += self.sdf_loss_term(pred_sdf, gt_sdf)

            if self.opt.use_xyz:
                loss_dict['xyz_loss'] += self.xyz_loss_term(preds[:,1:,:], self.norm_gt_xyzs, self.gt_xyz_mask)

        loss_dict['sdf_loss'] /= len(self.intermediate_preds_list)
        loss_dict['total_loss'] += loss_dict['sdf_loss']
        if self.opt.use_xyz:
            loss_dict['xyz_loss'] /= len(self.intermediate_preds_list)
            loss_dict['total_loss'] += self.opt.xyz_lambda * loss_dict['xyz_loss']

        return loss_dict

    def forward(self, images, points, calibs, labels=None, transforms=None, gt_xyzs=None, gt_xyz_mask=None, pairwise_dist=None, pairwise_pt_idxs=None):
        # pdb.set_trace()
        if self.opt.use_xyz:
            # norm_xyz_factor = max(self.opt.bbx_size) / 2
            # self.norm_points_model = xyzs / norm_xyz_factor
            self.norm_gt_xyzs = gt_xyzs
            self.gt_xyz_mask = gt_xyz_mask

        # Get image feature
        self.filter(images)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels)

        # get the prediction
        res = self.get_preds()

        # get the error
        loss_dict = self.get_loss()

        if self.opt.use_xyz:
            return res, loss_dict, self.xyzs, self.uvz
        else:
            return res, loss_dict, self.uvz
