import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Experiment launch: Logistic/Datasets related
        g_logistic = parser.add_argument_group('Logistic')
        g_logistic.add_argument('--exp_id', type=str, default='ncf_ycbv_run2',help='')
        g_logistic.add_argument('--work_base_path', type=str, default='/data1/lin/ncf_results/runs',help='')
        
        g_logistic.add_argument('--dataset', type=str, default='ycbv',help='lm | lmo | ycbv')
        g_logistic.add_argument('--train_data', type=str, default='ycbv', help='lm | ycbv | ycbv_real')
        g_logistic.add_argument('--more_train_data', type=str, default='none', help='ycbv_real')
        g_logistic.add_argument('--eval_data', type=str, default='ycbv_bop_cha', help='lm_bop_cha | lmo_bop_cha | ycbv_bop_cha')
        g_logistic.add_argument('--model_dir', type=str, default='/data2/lin/bop_datasets/ycbv/models', help='')
        g_logistic.add_argument('--ds_lm_dir', type=str, default='/data2/lin/bop_datasets/lm', help='')
        g_logistic.add_argument('--ds_lmo_dir', type=str, default='/data2/lin/bop_datasets/lmo', help='')
        g_logistic.add_argument('--ds_ycbv_dir', type=str, default='/data2/lin/bop_datasets/ycbv', help='')

        g_logistic.add_argument('--visib_fract_thresh', type=float, default=0.3, help='0.05 | 0.1 | 0.15 | 0.3')
        g_logistic.add_argument('--model_unit', type=str, default='mm', help='meter | mm')

        g_logistic.add_argument('--obj_id', default=2, type=int, help='ids for object')
        g_logistic.add_argument('--wks_size', type=int, default=[1600, 1600, 2000], help='size of workspace/mm')
        g_logistic.add_argument('--wks_z_shift', type=int, default=1010, help='shift of workspace/mm')
        g_logistic.add_argument('--test_wks_size', type=int, default=[1200, 1200, 930], help='size of test workspace/mm')
        g_logistic.add_argument('--test_wks_z_shift', type=int, default=925, help='shift of test workspace/mm')
        g_logistic.add_argument('--max_sym_disc_step', type=float, default=0.01, help='')
        g_logistic.add_argument('--sample_ratio', type=int, default=20, help='20 | 24 | 16 | 32 for surf')
        g_logistic.add_argument('--bbx_size', type=int, default=380, help='size of object bounding box/mm')
        g_logistic.add_argument('--bbx_shift', type=int, default=0, help='shift of object bounding box/mm')
        g_logistic.add_argument('--use_remap', type=bool, default=True, help='')
        g_logistic.add_argument('--rdist_norm', type=str, default='uvf', help='normlization method for ray distance, uvf|minmax')

        g_logistic.add_argument('--img_size', type=int, default=[640,480], help='image shape')
        g_logistic.add_argument('--num_views', type=int, default=1, help='How many views to use for multiview network.')

        g_logistic.add_argument('--GPU_ID', default=[0], type=int, help='# of GPUs')
        g_logistic.add_argument('--deterministic', type=bool, default=False, help='')
        g_logistic.add_argument('--seed', type=int, default=0)

        g_logistic.add_argument('--continue_train', type=bool, default=False, help='continue training: load model')
        g_logistic.add_argument('--resume_epoch', type=int, default=0, help='epoch resuming the training')
        g_logistic.add_argument('--eval_perf', type=bool, default=False, help='evaluation: load model')
        g_logistic.add_argument('--eval_epoch', type=int, default=0, help='epoch for eval.')

        g_logistic.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        g_logistic.add_argument('--load_optG_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        g_logistic.add_argument('--name', type=str, default='example',
                           help='name of the experiment. It decides where to store/load samples and models')

        # Sampling related
        g_sample = parser.add_argument_group('Sampling')
        g_sample.add_argument('--sigma_ratio', type=float, default=0.5, help='perturbation ratio of standard deviation for positions: 0.5 | 0.75')

        g_sample.add_argument('--num_sample_inout', type=int, default=5000, help='# of sampling points: 5000')

        # Rigid pose related
        g_rigid = parser.add_argument_group('Rigid')
        g_rigid.add_argument('--min_samples', type=int, default=3, help='min. #samples for ransac')
        g_rigid.add_argument('--res_thresh', type=float, default=20, help='residual threshold for selecting inliers')
        g_rigid.add_argument('--max_trials', type=int, default=200, help='max. #iterations')

        # Pre. & Aug. related
        g_aug = parser.add_argument_group('aug')
        # appearance
        g_aug.add_argument('--use_aug', type=bool, default=True, help='')
        g_aug.add_argument('--aug_blur', type=int, default=3, help='augmentation blur')
        g_aug.add_argument('--aug_sha', type=float, default=50.0, help='augmentation sharpness')
        g_aug.add_argument('--aug_con', type=float, default=50.0, help='augmentation contrast')
        g_aug.add_argument('--aug_bri', type=float, default=6.0, help='augmentation brightness')
        g_aug.add_argument('--aug_col', type=float, default=20.0, help='augmentation color')

        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--batch_size', type=int, default=4, help='input batch size')

        g_train.add_argument('--num_threads', default=1, type=int, help='# sthreads for loading data')
        g_train.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')
        # g_train.add_argument('--pin_memory', type=bool, default=True, help='pin_memory')

        g_train.add_argument('--out_type', type=str, default='rsdf', help='rsdf | csdf | eff_csdf')
        g_train.add_argument('--loss_type', type=str, default='l1', help='mse | l1 | huber')
        g_train.add_argument('--clamp_dist', type=float, default=5.0, help='')
        g_train.add_argument('--norm_clamp_dist', type=float, default=0.1, help='')
        g_train.add_argument('--use_xyz', type=bool, default=True, help='')
        g_train.add_argument('--xyz_lambda', type=float, default=1.0, help='')

        g_train.add_argument('--init_type', type=str, default='normal', help='normal | xavier | kaiming | orthogonal')
        g_train.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal')
        g_train.add_argument('--optimizer', choices=["adam", "rms"], default="rms")
        g_train.add_argument('--learning_rate', type=float, default=1e-4, help='') # 1e-3
        g_train.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        g_train.add_argument('--schedule', type=int, nargs='+', default=[500, 1000, 1500],
                            help='Decrease learning rate at these epochs.') 
        g_train.add_argument('--num_epoch', type=int, default=2000, help='num epoch to train')

        g_train.add_argument('--freq_plot', type=int, default=7000, help='freqency of the error plot')
        g_train.add_argument('--freq_debug', type=int, default=7000, help='frequence of the visualization')

        # Model related
        g_model = parser.add_argument_group('Model')
        # General
        g_model.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')
        # hg filter specify
        g_model.add_argument('--num_stack', type=int, default=4, help='# of stacked layer of hourglass')
        g_model.add_argument('--num_hourglass', type=int, default=2, help='# of hourglass')
        g_model.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        g_model.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        g_model.add_argument('--hourglass_dim', type=int, default='256', help='256 | 512')

        # Classification General
        g_model.add_argument('--mlp_dim', nargs='+', default=[257, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp')
        g_model.add_argument('--mlp_dim_xyz', nargs='+', default=[257, 1024, 512, 256, 128, 4],
                             type=int, help='# of dimensions of mlp')

        g_model.add_argument('--use_tanh', type=bool, default=True,
                             help='using tanh after last conv of image_filter network')

        g_model.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')

        # Eval. related
        g_eval = parser.add_argument_group('Evaluation')
        g_eval.add_argument('--step_size', type=int, default=10, help='step size (mm) of grid')
        g_eval.add_argument('--num_in_batch', type=int, default=1500000, help='number of each batch for eval.')
        g_eval.add_argument('--thresh', type=float, default=0.0, help='0.0999 | 0.0 | -0.0999')

        g_eval.add_argument('--freq_eval_all', type=int, default=20, help='freqency of the eval. for all')
        g_eval.add_argument('--gen_obj_pose', type=bool, default=True, help='')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
