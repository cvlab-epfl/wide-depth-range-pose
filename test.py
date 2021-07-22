import os
import sys
import time
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

import numpy as np
import cv2

from argument import get_args
from backbone import darknet53
from dataset import BOP_Dataset, collate_fn
from model import PoseModule
import transform
from evaluate import evaluate
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    DistributedSampler,
    all_gather,
)
from train import (
    accumulate_dicts,
    valid,
    data_sampler,
)
from utils import (
    visualize_accuracy_per_depth,
    print_accuracy_per_class,
)

from tensorboardX import SummaryWriter

# reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
np.random.seed(0)

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

    cfg = get_args()

    # create working_dir dynamically
    timestr = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
    name_wo_ext = os.path.splitext(os.path.split(cfg['RUNTIME']['CONFIG_FILE'])[1])[0]
    working_dir = 'working_dirs' + '/' + name_wo_ext + '/' + timestr + '/'
    cfg['RUNTIME']['WORKING_DIR'] = working_dir
    print("working directory: " + cfg['RUNTIME']['WORKING_DIR'])
    if get_rank() == 0:
        os.makedirs(cfg['RUNTIME']['WORKING_DIR'], exist_ok=True)
        logger = SummaryWriter(cfg['RUNTIME']['WORKING_DIR'])

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    cfg['RUNTIME']['N_GPU'] = n_gpu
    cfg['RUNTIME']['DISTRIBUTED'] = n_gpu > 1

    if cfg['RUNTIME']['DISTRIBUTED']:
        torch.cuda.set_device(cfg['RUNTIME']['LOCAL_RANK'])
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        synchronize()

    # device = 'cuda'
    device = cfg['RUNTIME']['RUNNING_DEVICE']

    internal_K = np.array(cfg['INPUT']['INTERNAL_K']).reshape(3,3)

    valid_trans = transform.Compose(
        [
            transform.Resize(
                cfg['INPUT']['INTERNAL_WIDTH'], 
                cfg['INPUT']['INTERNAL_HEIGHT'], 
                internal_K),
            transform.Normalize(
                cfg['INPUT']['PIXEL_MEAN'], 
                cfg['INPUT']['PIXEL_STD']),
            transform.ToTensor(), 
        ]
    )

    valid_set = BOP_Dataset(
        cfg['DATASETS']['TEST'],
        cfg['DATASETS']['MESH_DIR'], 
        cfg['DATASETS']['BBOX_FILE'], 
        valid_trans,
        training = False)

    if cfg['MODEL']['BACKBONE'] == 'darknet53':
        backbone = darknet53(pretrained=False)
    else:
        print("unsupported backbone!")
        assert(0)
    model = PoseModule(cfg, backbone)

    # load weight
    if os.path.exists(cfg['RUNTIME']['WEIGHT_FILE']):
        try:
            chkpt = torch.load(cfg['RUNTIME']['WEIGHT_FILE'], map_location='cpu')  # load checkpoint
            if 'model' in chkpt:
                chkpt = chkpt['model']
            # model.load_state_dict(chkpt) # strict
            # loose loading
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in chkpt.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            # 
            print('Weights are loaded from ' + cfg['RUNTIME']['WEIGHT_FILE'])
        except:
            print('Loading weights from %s is failed' % (cfg['RUNTIME']['WEIGHT_FILE']))
            print("Random initialized weights.")
    else:
        print("Random initialized weights.")

    model = model.to(device)
    
    batch_size_per_gpu = int(cfg['TEST']['IMS_PER_BATCH'] / cfg['RUNTIME']['N_GPU'])
    if batch_size_per_gpu == 0:
        print('ERROR: %d GPUs for %d batch(es)' % (cfg['RUNTIME']['N_GPU'], cfg['TEST']['IMS_PER_BATCH']))
        assert(0)

    if cfg['RUNTIME']['DISTRIBUTED']:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg['RUNTIME']['LOCAL_RANK']],
            output_device=cfg['RUNTIME']['LOCAL_RANK'],
            broadcast_buffers=False,
        )
        model = model.module
 
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size_per_gpu,
        sampler=data_sampler(valid_set, shuffle=False, distributed=cfg['RUNTIME']['DISTRIBUTED']),
        num_workers=cfg['RUNTIME']['NUM_WORKERS'],
        collate_fn=collate_fn(cfg['INPUT']['SIZE_DIVISIBLE']),
    )

    accuracy_adi_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range = \
        valid(cfg, 0, valid_loader, model, device, logger=logger)

    visImg = visualize_accuracy_per_depth(
        accuracy_adi_per_class, 
        accuracy_rep_per_class, 
        accuracy_adi_per_depth, 
        accuracy_rep_per_depth, 
        depth_range)

    visFileName = cfg['RUNTIME']['WORKING_DIR'] + 'error_statistics_per_depth.png'
    cv2.imwrite(visFileName, visImg)
    print("Error statistics for each depth bin are saved to '%s'" % visFileName)
