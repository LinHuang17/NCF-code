#!/usr/bin/env python3
# Copyright 2020-present Zerong Zheng. All Rights Reserved.

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


"""
XYZ loss w/o symmetry
"""
class XYZLoss(nn.Module):
    def __init__(self, use_xyz_mask=True):
        super(XYZLoss, self).__init__()
        # self.criterion = nn.MSELoss(reduction='mean')
        self.criterion = nn.SmoothL1Loss()
        self.use_xyz_mask = use_xyz_mask

    def forward(self, output, target, xyz_mask):
        '''
        should be consistent for tensor shape
        (B, 3, N)/(B, 3, N)/(B, 1, N) or
        (B, N, 3)/(B, N, 3)/(B, N, 1)
        '''
        if self.use_xyz_mask:
            loss = self.criterion(
                output.mul(xyz_mask),
                target.mul(xyz_mask)
            )
        else:
            # loss += 0.5 * self.criterion(xyz_pred, xyz_gt)
            loss = self.criterion(output, target)

        return loss

"""
XYZ loss with symmetry
"""
class XYZLoss_sym(nn.Module):
    def __init__(self, use_xyz_mask=True, sym_pool=None):
        super(XYZLoss_sym, self).__init__()
        # self.criterion = nn.MSELoss(reduction='mean')
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.use_xyz_mask = use_xyz_mask

        self.sym_pool = sym_pool

    def forward(self, output, target, xyz_mask):
        '''
        should be consistent for tensor shape
        (B, 3, N)/(B, 3, N)/(B, 1, N) or
        (B, N, 3)/(B, N, 3)/(B, N, 1)
        '''
        output = output.permute(0,2,1)
        target = target.permute(0,2,1)
        xyz_mask = xyz_mask.permute(0,2,1)
        if (len(self.sym_pool) > 1):
            for sym_id, transform in enumerate(self.sym_pool):
                # repeat: (3, 3) -> (B, 3, 3)
                rot = transform[:3, :3].cuda().repeat((target.size(0),1,1))
                # repeat: (3, 1) -> (B, 3, 1)
                trans = transform[:3, 3:4].cuda().repeat((target.size(0),1,1))
                # (B, 3, 3) * (B, 3, N)  + (B, 3, 1) -> (B, 3, N) -> (B, N, 3)
                sym_target = torch.baddbmm(trans, rot, target.permute(0,2,1)).permute(0,2,1)
                if self.use_xyz_mask:
                    # (B, N, 3)
                    loss_xyz_temp = self.criterion(output.mul(xyz_mask), sym_target.mul(xyz_mask))
                else:
                    # loss += 0.5 * self.criterion(xyz_pred, xyz_gt)
                    # (B, N, 3)
                    loss_xyz_temp = self.criterion(output, sym_target)
                # (B, N)
                loss_xyz_temp = torch.sum(loss_xyz_temp, dim=2) / 3
                # (B)
                loss_sum = torch.sum(loss_xyz_temp, dim=1)
                if(sym_id > 0):
                    # (M, B)
                    loss_sums = torch.cat((loss_sums, loss_sum.unsqueeze(0)), dim=0)
                    # (M, B, N)
                    loss_xyzs = torch.cat((loss_xyzs, loss_xyz_temp.unsqueeze(0)), dim=0)
                else:
                    loss_sums = loss_sum.unsqueeze(0)
                    loss_xyzs = loss_xyz_temp.unsqueeze(0)
            # (1, B)
            min_values = torch.min(loss_sums, dim=0, keepdim=True)[0]
            # (M, B)
            loss_switch = torch.eq(loss_sums, min_values).type(output.dtype)
            # (M, B, 1) * (M, B, N) -> (M, B, N)
            loss_xyz = loss_switch.unsqueeze(2) * loss_xyzs
            # (B, N)
            loss_xyz = torch.sum(loss_xyz, dim=0)
        else:
            if self.use_xyz_mask:
                # (B, N, 3)
                loss_xyz = self.criterion(output.mul(xyz_mask), target.mul(xyz_mask))
            else:
                # (B, N, 3)
                loss_xyz = self.criterion(output, target)
            # (B, N)
            loss_xyz = torch.sum(loss_xyz, dim=2) / 3
        loss = loss_xyz
        loss = torch.mean(loss)

        return loss

class XYZLoss_old(nn.Module):
    def __init__(self, use_xyz_mask=True):
        super(XYZLoss_orig, self).__init__()
        # self.criterion = nn.MSELoss(reduction='mean')
        self.criterion = nn.SmoothL1Loss()
        self.use_xyz_mask = use_xyz_mask

    def forward(self, output, target, use_xyz_mask):
        batch_size = output.size(0)
        num_queries = output.size(1)
        xyzs_pred = output.reshape((batch_size, num_queries, -1)).split(1, 1)
        xyzs_gt = target.reshape((batch_size, num_queries, -1)).split(1, 1)
        loss = 0

        for idx in range(num_queries):
            xyz_pred = xyzs_pred[idx].squeeze()
            xyz_gt = xyzs_gt[idx].squeeze()
            if self.use_xyz_mask:
                # loss += 0.5 * self.criterion(
                loss += self.criterion(
                    xyz_pred.mul(use_xyz_mask[:, idx]),
                    xyz_gt.mul(use_xyz_mask[:, idx])
                )
            else:
                # loss += 0.5 * self.criterion(xyz_pred, xyz_gt)
                loss +=self.criterion(xyz_pred, xyz_gt)

        return loss / num_queries


class LipschitzLoss(nn.Module):
    def __init__(self, k, reduction=None):
        super(LipschitzLoss, self).__init__()
        self.relu = nn.ReLU()
        self.k = k
        self.reduction = reduction

    def forward(self, x1, x2, y1, y2):
        l = self.relu(torch.norm(y1-y2, dim=-1) / (torch.norm(x1-x2, dim=-1)+1e-3) - self.k)
        # l = torch.clamp(l, 0.0, 5.0)    # avoid
        if self.reduction is None or self.reduction == "mean":
            return torch.mean(l)
        else:
            return torch.sum(l)


class HuberFunc(nn.Module):
    def __init__(self, reduction=None):
        super(HuberFunc, self).__init__()
        self.reduction = reduction

    def forward(self, x, delta):
        n = torch.abs(x)
        cond = n < delta
        l = torch.where(cond, 0.5 * n ** 2, n*delta - 0.5 * delta**2)
        if self.reduction is None or self.reduction == "mean":
            return torch.mean(l)
        else:
            return torch.sum(l)


class SoftL1Loss(nn.Module):
    def __init__(self, reduction=None):
        super(SoftL1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, eps=0.0, lamb=0.0):
        ret = torch.abs(input - target) - eps
        ret = torch.clamp(ret, min=0.0, max=100.0)
        ret = ret * (1 + lamb * torch.sign(target) * torch.sign(target-input))
        if self.reduction is None or self.reduction == "mean":
            return torch.mean(ret)
        else:
            return torch.sum(ret)



if __name__ == '__main__':

    criterion1 = XYZLoss()
    criterion2 = XYZLoss_orig()
    aa = torch.rand((2,5000,3))
    bb = torch.rand((2,5000,3))
    # bb = aa.clone()
    mask = torch.rand((2,5000,1)) > 0.3
    pdb.set_trace()

    loss1 = criterion1(aa,bb,mask)
    loss2 = criterion1(aa.permute(0,2,1),bb.permute(0,2,1),mask.permute(0,2,1))
    loss3 = criterion2(aa,bb,mask)

    print('debug')
