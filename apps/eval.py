"""
 eval. for rigid obj.
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
from lib.eval_Rt_time_util import *
from lib.options import BaseOptions


# get options
opt = BaseOptions().parse()


def evaluate(opt):

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

    # set gpu environment
    devices_ids = opt.GPU_ID
    num_GPU = len(devices_ids)
    torch.cuda.set_device(devices_ids[0])

    # dataset
    test_dataset_list = []
    test_data_ids = [opt.eval_data]
    for data_id in test_data_ids:
        if data_id == 'lm_bop_cha':
            test_dataset_list.append(BOP_BP_LM(opt, phase='test'))
        if data_id == 'lmo_bop_cha':
            test_dataset_list.append(BOP_BP_LMO(opt, phase='test'))
        if data_id == 'ycbv_bop_cha':
            test_dataset_list.append(BOP_BP_YCBV(opt, phase='test'))
    projection_mode = test_dataset_list[0].projection_mode
    test_dataset = ConcatDataset(test_dataset_list)
    # create test data loader
    # NOTE: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=opt.num_threads, pin_memory=(opt.num_threads == 0))
                                #   persistent_workers=(opt.num_threads > 0))
                                #   num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_dataset))

    # define model, multi-gpu, checkpoint
    sdf_criterion = None
    xyz_criterion = None
    netG = HGPIFuNet(opt, projection_mode,
                     sdf_loss_term=sdf_criterion,
                     xyz_loss_term=xyz_criterion)
    print('Using Network: ', netG.name)

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

    # evaluation
    with torch.no_grad():
        set_eval()
        obj_id = [opt.obj_id][0]
        print('eval. for obj. pose and time (test) ...')
        save_csv_path = os.path.join(results_path, opt.name, f'ncf-obj{obj_id}_{opt.dataset}-Rt-time.csv')
        eval_Rt_time(opt, netG.module, test_data_loader, save_csv_path)

if __name__ == '__main__':
    evaluate(opt)
