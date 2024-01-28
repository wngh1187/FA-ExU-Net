import os
from itertools import chain
import torch

def get_args():
    system_args = {
        # expeirment info
        'project'       : 'SVSN',
        'name'          : 'FA_ExU_Net_Medium',
        'tags'          : ['Proposed'],
        'description'   : '',
        'module_name'   : 'FA_ExU_Net_Bottel',

        # local
        'path_logging'  : '/results',

        # VoxCeleb1 DB
        'path_vox1_train'   : '/data/Voxcelebs/VoxCeleb1/train',
        'path_vox1_test'    : '/data/Voxcelebs/VoxCeleb1/test',
        'path_vox1_trials'  : '/data/Voxcelebs/VoxCeleb1/trials.txt',
        'path_vox2_train'   : '/data/Voxcelebs/VoxCeleb2',

        # musan DB
        'path_musan'        : '/datas/musan',

        # device
        'num_workers'   : 16,
        'usable_gpu'    : '0,1,2,3,4,5',
        'tqdm_ncols'    : 90,
        'path_scripts'     : os.path.dirname(os.path.realpath(__file__))
    }
    
    experiment_args = {
        # env
        'epoch'                     : 188,
        'batch_size'                : 320,
        'number_iteration_for_log'  : 50,
        'rand_seed'                 : 1234,
        'flag_reproduciable'        : True,
        
        # train process
        'do_train_feature_enhancement'  : True,
        'do_train_code_enhancement'     : True,
        
        # optimizer
        'optimizer'                 : 'adam',
        'amsgrad'                   : True,
        'learning_rate_scheduler'   : 'warmup',
        'lr_start'                  : 1e-7,
        'lr_end'                    : 1e-7,
        'number_cycle'              : 40,
        'warmup_number_cycle'       : 1,
        'T_mult'                    : 1.5,
        'eta_max'                   : 1e-2,
        'gamma'                     : 0.5,
        'weigth_decay'              : 1e-4,

        # criterion
        'classification_loss'                               : 'aam_softmax',
        'enhancement_loss'                                  : 'mse',
        'code_enhacement_loss'                              : 'angleproto',
        'weight_classification_loss'                        : 1,
        'weight_code_enhancement_loss'                      : 1,
        'weight_feature_enhancement_loss'                   : 1,

        # model
        'block'                 : 'bottle',
        'first_kernel_size'     : 7,
        'first_stride_size'     : (2,1),
        'first_padding_size'    : 3,
        'l_channel'             : [64, 64, 128, 128, 256, 256],
        'l_num_convblocks'      : [1, 1, 1, 2, 2, 1],
        'stride'                : [1,1,1,1,1,1],
        'code_dim'              : 128,
        
        

        # data
        'nb_utt_per_spk'    : 2, 
        'max_seg_per_spk'   : 500,
        'winlen'            : 400,
        'winstep'           : 160,
        'train_frame'       : 382,
        'nfft'              : 1024,
        'samplerate'        : 16000,
        'nfilts'            : 64,
        'premphasis'        : 0.97,
        'winfunc'           : torch.hamming_window,
        'test_frame'        : 382,
        'spec_mask_F'       : 8,
        'spec_mask_T'       : 40,
        'aam_margin'        : 0.3,
        'aam_scale'         : 30,

    }

    # set args (system_args + experiment_args)
    args = {}
    for k, v in chain(system_args.items(), experiment_args.items()):
        args[k] = v

    return args, system_args, experiment_args