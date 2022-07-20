"""
 train & eval. for rigid obj.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import cv2
import json
import time
import random
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from lib.data import *
from lib.model import *
from lib.net_util import *
from lib.sym_util import *
from lib.loss_util import *
from lib.eval_Rt_time_util import *
from lib.options import BaseOptions

from lib.debug_pyrender_util import *
from torch.utils.tensorboard import SummaryWriter

# get options
opt = BaseOptions().parse()

class meter():

    def __init__(self, opt):

        self.opt = opt

        self.load_time = AverageMeter()
        self.forward_time = AverageMeter()

        self.sdf_loss_meter = AverageMeter()
        if self.opt.use_xyz:
            self.xyz_loss_meter = AverageMeter()
        self.total_loss_meter = AverageMeter()

    def update_time(self, time, end, state):

        if state == 'forward':
            self.forward_time.update(time - end)
        elif state == 'load':
            self.load_time.update(time - end)

    def update_total_loss(self, total_loss, size):

        self.total_loss_meter.update(total_loss.item(), size)

    def update_loss(self, loss_dict, size):

        self.sdf_loss_meter.update(loss_dict['sdf_loss'].mean().item(), size)
        if self.opt.use_xyz:
            self.xyz_loss_meter.update(loss_dict['xyz_loss'].mean().item(), size)

def set_dataset_train_mode(dataset, mode=True):
    for dataset_idx in range(len(dataset.datasets)):
        dataset.datasets[dataset_idx].is_train = mode

def train(opt):

    # seed
    if opt.deterministic:
        seed = opt.seed
        print("Set manual random Seed: ", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) # cpu
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmarking enabled")

    # set path
    work_path = os.path.join(opt.work_base_path, f"{opt.exp_id}")
    os.makedirs(work_path, exist_ok=True)
    checkpoints_path = os.path.join(work_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    results_path = os.path.join(work_path, "results")
    os.makedirs(results_path, exist_ok=True)
    tb_dir = os.path.join(work_path, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    tb_runs_dir = os.path.join(tb_dir, "runs")
    os.makedirs(tb_runs_dir, exist_ok=True)
    debug_dir = os.path.join(work_path, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(tb_runs_dir, f'{opt.exp_id}'))
    writer.add_text('Info', 'ncf for obj. Rt est. in frustum space using pred. sdf & xyz')

    # set gpu environment
    devices_ids = opt.GPU_ID
    num_GPU = len(devices_ids)
    torch.cuda.set_device(devices_ids[0])

    # dataset
    train_dataset_list = []
    train_data_ids = [opt.train_data] + [opt.more_train_data]
    for data_id in train_data_ids:
        if data_id == 'lm':
            train_dataset_list.append(BOP_BP_LM(opt, phase='train'))
        if data_id == 'ycbv':
            train_dataset_list.append(BOP_BP_YCBV(opt, phase='train'))
        if data_id == 'ycbv_real':
            train_dataset_list.append(BOP_BP_YCBV_real(opt, phase='train'))
    projection_mode = train_dataset_list[0].projection_mode
    train_dataset = ConcatDataset(train_dataset_list)
    # create train data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=(opt.num_threads == 0))
                                #    persistent_workers=(opt.num_threads > 0))
                                #    num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('train data size: ', len(train_dataset))

    test_dataset_list = []
    test_data_ids = [opt.eval_data]
    for data_id in test_data_ids:
        if data_id == 'lm_bop_cha':
            test_dataset_list.append(BOP_BP_LM(opt, phase='test'))
        if data_id == 'lmo_bop_cha':
            test_dataset_list.append(BOP_BP_LMO(opt, phase='test'))
        if data_id == 'ycbv_bop_cha':
            test_dataset_list.append(BOP_BP_YCBV(opt, phase='test'))
    test_dataset = ConcatDataset(test_dataset_list)
    # create test data loader
    # NOTE: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=opt.num_threads, pin_memory=(opt.num_threads == 0))
                                #   persistent_workers=(opt.num_threads > 0))
                                #   num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_dataset))

    # pre-define pool of symmetric poses
    sym_pool=[]
    obj_id = [opt.obj_id][0]
    # load obj. param.
    obj_params = get_obj_params(opt.model_dir, [opt.train_data][0])
    # Load meta info about the models (including symmetries).
    models_info = load_json(obj_params['models_info_path'], keys_to_int=True)
    sym_poses = get_symmetry_transformations(models_info[obj_id], opt.max_sym_disc_step)
    for sym_pose in sym_poses:
        Rt = np.concatenate([sym_pose['R'], sym_pose['t'].reshape(3,1)], axis=1)
        Rt = np.concatenate([Rt, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
        sym_pool.append(torch.Tensor(Rt))

    # define model, multi-gpu, checkpoint
    if opt.loss_type == 'mse':
        sdf_criterion = torch.nn.MSELoss()
    elif opt.loss_type == 'l1':
        sdf_criterion = torch.nn.L1Loss()
    elif opt.loss_type == 'huber':
        sdf_criterion = torch.nn.SmoothL1Loss()
    xyz_criterion = None
    if opt.use_xyz:
        if (len(sym_pool) > 1):
            xyz_criterion = XYZLoss_sym(use_xyz_mask=True, sym_pool=sym_pool)
        else:
            xyz_criterion = XYZLoss(use_xyz_mask=True)
    netG = HGPIFuNet(opt, projection_mode,
                     sdf_loss_term=sdf_criterion,
                     xyz_loss_term=xyz_criterion)
    print('Using Network: ', netG.name)

    def set_train():
        netG.train()

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.continue_train or opt.eval_perf:
        print('Loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=torch.device('cpu')))

    # Data Parallel
    # if num_GPU > 1:
    netG = torch.nn.DataParallel(netG, device_ids=devices_ids, output_device=devices_ids[0])
    # netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=devices_ids, output_device=devices_ids[0])
    print(f'Data Paralleling on GPU: {devices_ids}')
    netG.cuda()

    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs('%s/%s' % (checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (results_path, opt.name), exist_ok=True)
    opt_log = os.path.join(results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    # optimizer
    lr = opt.learning_rate
    if opt.optimizer == 'rms':
        optimizerG = torch.optim.RMSprop(netG.module.parameters(), lr=lr, momentum=0, weight_decay=0)
        print(f'Using optimizer: rms')
        # optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr, momentum=0, weight_decay=0)
        # optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        optimizerG = torch.optim.Adam(netG.module.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
        print(f'Using optimizer: adam')
        # optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # load optimizer
    if opt.continue_train and opt.load_optG_checkpoint_path is not None:
        print('Loading for opt G ...', opt.load_optG_checkpoint_path)
        optimizerG.load_state_dict(torch.load(opt.load_optG_checkpoint_path))

    # training
    tb_train_idx = 0
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch,0)
    for epoch in range(start_epoch, opt.num_epoch):
        # log lr
        writer.add_scalar('train/learning_rate', lr, epoch)

        # meter, time, train mode
        train_meter = meter(opt)
        epoch_start_time = time.time()
        set_train()
        # torch.cuda.synchronize()
        iter_data_time = time.time()
        for train_idx, train_data in enumerate(train_data_loader):
            tb_train_idx += 1
            # measure elapsed data loading time in batch
            iter_start_time = time.time()
            train_meter.update_time(iter_start_time, iter_data_time, 'load')

            # retrieve the data
            # shape (B, 3, 480, 640)
            image_tensor = train_data['img'].cuda()
            # shape (B, 4, 4)
            calib_tensor = train_data['calib'].cuda()
            # shape (B, 3, 5000)
            sample_tensor = train_data['samples'].cuda()
            batch = image_tensor.size(0)

            # shape (B, 1, 5000)
            label_tensor = train_data['labels'].cuda()
            if opt.use_xyz:
                # shape (B, 1, 5000)
                xyz_tensor = train_data['xyzs'].cuda()
                xyz_mask_tensor = train_data['xyz_mask'].cuda()
            transforms = torch.zeros([batch,2,3]).cuda()
            transforms[:, 0,0] = 1 / (opt.img_size[0] // 2)
            transforms[:, 1,1] = 1 / (opt.img_size[1] // 2)
            transforms[:, 0,2] = -1
            transforms[:, 1,2] = -1
            if opt.use_xyz:
                results, loss_dict, xyzs, uvz = netG(image_tensor, sample_tensor, calib_tensor,
                                                     labels=label_tensor, transforms=transforms,
                                                     gt_xyzs=xyz_tensor, gt_xyz_mask=xyz_mask_tensor)
            else:
                results, loss_dict, uvz = netG(image_tensor, sample_tensor, calib_tensor,
                                               labels=label_tensor, transforms=transforms)

            optimizerG.zero_grad()
            # for param in netG.module.parameters():
            # for param in netG.parameters():
                # param.grad = None
            loss_dict['total_loss'].mean().backward()
            # error.backward()
            optimizerG.step()

            # measure elapsed forward time in batch
            # torch.cuda.synchronize()
            iter_net_time = time.time()
            train_meter.update_time(iter_net_time, iter_start_time, 'forward')
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            # update loss
            train_meter.update_loss(loss_dict, batch)
            # update total loss
            train_meter.update_total_loss(loss_dict['total_loss'].mean(), batch)

            writer.add_scalar('train/total_loss_per_batch', train_meter.total_loss_meter.val, tb_train_idx)
            writer.add_scalar('train/sdf_loss_per_batch', train_meter.sdf_loss_meter.val, tb_train_idx)
            if opt.use_xyz:
                writer.add_scalar('train/xyz_loss_per_batch', train_meter.xyz_loss_meter.val, tb_train_idx)
            if train_idx % opt.freq_plot == 0:
                print('Name: {0} | Epoch: {1} | {2}/{3} | Loss: {4:.06f} | LR: {5:.06f} | dataT: {6:.05f} | netT: {7:.05f} | ETA: {8:02d}:{9:02d}'.format(
                      opt.name, epoch, train_idx, len(train_data_loader), loss_dict['total_loss'].mean().item(), lr,
                      iter_start_time - iter_data_time, iter_net_time - iter_start_time, int(eta // 60), int(eta - 60 * (eta // 60))))

            if train_idx % opt.freq_debug == 0:
                with torch.no_grad():
                    # debug for rgb, mask, rendering of object model
                    # shape (4, 3, 480, 640)
                    name = train_data['name'][0]
                    model_mesh = train_data_loader.dataset.datasets[0].model_mesh_dict[name].copy(include_cache=False)
                    img = (np.transpose(image_tensor[0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)
                    save_debug_path = os.path.join(debug_dir, f'train_sample{train_idx}_epoch{epoch}_debug.jpeg')
                    viz_debug_data(img, model_mesh,
                                   train_data['extrinsic'][0].detach().numpy(), train_data['aug_intrinsic'][0].detach().numpy(),
                                   save_debug_path)

                    # debug for query projection during forward
                    # shape (4, 3, 5000), (4, 1, 5000)
                    inv_trans = torch.zeros([1,2,3])
                    inv_trans[:, 0,0] = (opt.img_size[0] // 2)
                    inv_trans[:, 1,1] = (opt.img_size[1] // 2)
                    inv_trans[:, 0,2] = (opt.img_size[0] // 2)
                    inv_trans[:, 1,2] = (opt.img_size[1] // 2)
                    scale = inv_trans[:, :2, :2]
                    shift = inv_trans[:, :2, 2:3]
                    uv = torch.baddbmm(shift, scale, uvz[0].detach().cpu()[:2, :].unsqueeze(0))
                    query_res = {'img': image_tensor[0].detach().cpu(), 'samples': uv.squeeze(0), 'labels': label_tensor[0].detach().cpu()}
                    save_in_query_path = os.path.join(debug_dir, f'train_sample{train_idx}_epoch{epoch}_in_query.jpeg')
                    save_out_query_path = os.path.join(debug_dir, f'train_sample{train_idx}_epoch{epoch}_out_query.jpeg')
                    viz_debug_query_forward(opt.out_type, query_res, save_in_query_path, save_out_query_path)

                    # debug for prediction & gt ply for query & its label
                    save_gt_path = os.path.join(debug_dir, f'train_sample{train_idx}_epoch{epoch}_sdf_gt.ply')
                    save_sdf_path = os.path.join(debug_dir, f'train_sample{train_idx}_epoch{epoch}_sdf_est.ply')
                    r = results[0].cpu()
                    points = sample_tensor[0].transpose(0, 1).cpu()
                    if opt.out_type[-3:] == 'sdf':
                        save_samples_truncted_sdf(save_gt_path, points.detach().numpy(), label_tensor[0].transpose(0, 1).cpu().detach().numpy(), thres=opt.norm_clamp_dist)
                        save_samples_truncted_sdf(save_sdf_path, points.detach().numpy(), r.detach().numpy(), thres=opt.norm_clamp_dist)
                    if opt.use_xyz:
                        norm_xyz_factor = train_data['norm_xyz_factor'][0].item()
                        pred_xyzs = (xyzs[0].transpose(0, 1).cpu()) * norm_xyz_factor
                        save_sdf_xyz_path = os.path.join(debug_dir, f'train_sample{train_idx}_epoch{epoch}_xyz_est.ply')
                        save_samples_truncted_sdf(save_sdf_xyz_path, pred_xyzs.detach().numpy(), r.detach().numpy(), thres=opt.norm_clamp_dist)

            iter_data_time = time.time()

        writer.add_scalars('train/time_per_epoch', {'forward_per_batch': train_meter.forward_time.avg, 'dataload_per_batch': train_meter.load_time.avg}, epoch)
        writer.add_scalar('train/total_loss_per_epoch', train_meter.total_loss_meter.avg, epoch)
        writer.add_scalar('train/sdf_loss_per_epoch', train_meter.sdf_loss_meter.avg, epoch)
        if opt.use_xyz:
            writer.add_scalar('train/xyz_loss_per_epoch', train_meter.xyz_loss_meter.avg, epoch)
        # update learning rate
        lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)
        # save checkpoints
        torch.save(netG.module.state_dict(), '%s/%s/netG_epoch_%d' % (checkpoints_path, opt.name, epoch))
        torch.save(optimizerG.state_dict(), '%s/%s/optG_epoch_%d' % (checkpoints_path, opt.name, epoch))

        #### test
        with torch.no_grad():
            set_eval()
            obj_id = [opt.obj_id][0]
            if epoch > 0 and epoch % opt.freq_eval_all == 0 and opt.use_xyz and opt.gen_obj_pose:
                print('eval. for obj. pose and time (test) ...')
                save_csv_path = os.path.join(results_path, opt.name, f'objpifu-obj{obj_id}_{opt.dataset}-Rt-time.csv')
                eval_Rt_time(opt, netG.module, test_data_loader, save_csv_path)

    writer.close()

if __name__ == '__main__':
    train(opt)
