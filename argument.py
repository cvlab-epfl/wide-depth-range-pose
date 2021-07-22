import argparse
import yaml
import time
import os

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--config_file', type=str, default='./configs/swisscube.yaml')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--weight_file', type=str, default='')
    parser.add_argument('--running_device', type=str, default='cuda')

    return parser

def get_args():
    parser = get_argparser()
    args = parser.parse_args()

    # Read yaml configure
    with open(args.config_file, 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)

    cfg['RUNTIME'] = {}
    cfg['RUNTIME']['LOCAL_RANK'] = args.local_rank
    cfg['RUNTIME']['CONFIG_FILE'] = args.config_file
    cfg['RUNTIME']['NUM_WORKERS'] = args.num_workers
    cfg['RUNTIME']['WEIGHT_FILE'] = args.weight_file
    cfg['RUNTIME']['RUNNING_DEVICE'] = args.running_device
    
    if cfg['MODEL']['BACKBONE'] == 'darknet53':
        cfg['MODEL']['FEAT_CHANNELS'] = [0, 0, 256, 512, 1024]
    else:
        print('Unsupported backbone')
        assert(0)

    cfg['MODEL']['OUT_CHANNEL'] = 256
    cfg['MODEL']['N_CONV'] = 4
    cfg['MODEL']['PRIOR'] = 0.01
    if 'USE_HIGHER_LEVELS' not in cfg['MODEL']:
        cfg['MODEL']['USE_HIGHER_LEVELS'] = True

    cfg['SOLVER']['FOCAL_GAMMA'] = 2.0
    cfg['SOLVER']['FOCAL_ALPHA'] = 0.25
    cfg['SOLVER']['POSITIVE_NUM'] = 10

    cfg['INPUT']['PIXEL_MEAN'] = [0.485, 0.456, 0.406]
    cfg['INPUT']['PIXEL_STD'] = [0.229, 0.224, 0.225]
    cfg['INPUT']['SIZE_DIVISIBLE'] = 32

    if 'STEPS_PER_EPOCH' not in cfg['SOLVER']:
        cfg['SOLVER']['STEPS_PER_EPOCH'] = 1000

    return cfg
