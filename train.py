import os
import sys
import time
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import cv2
import math

from argument import get_args
from backbone import darknet53
from dataset import BOP_Dataset, collate_fn
from model import PoseModule
from scheduler import WarmupScheduler

import transform
from evaluate import evaluate
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    DistributedSampler,
    all_gather,
)
from utils import (
    load_bop_meshes,
    visualize_pred,
    print_accuracy_per_class,
)
from tensorboardX import SummaryWriter

# reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
np.random.seed(0)

# close shared memory of pytorch
if True:
    # https://github.com/huaweicloud/dls-example/issues/26
    from torch.utils.data import dataloader
    from torch.multiprocessing import reductions
    from multiprocessing.reduction import ForkingPickler
    default_collate_func = dataloader.default_collate
    def default_collate_override(batch):
        dataloader._use_shared_memory = False
        return default_collate_func(batch)
    setattr(dataloader, 'default_collate', default_collate_override)
    for t in torch._storage_classes:
        if sys.version_info[0] == 2:
            if t in ForkingPickler.dispatch:
                del ForkingPickler.dispatch[t]
        else:
            if t in ForkingPickler._extra_reducers:
                del ForkingPickler._extra_reducers[t]

def accumulate_dicts(data):
    all_data = all_gather(data)

    if get_rank() != 0:
        return

    data = {}

    for d in all_data:
        data.update(d)

    return data

@torch.no_grad()
def valid(cfg, epoch, loader, model, device, logger=None):
    torch.cuda.empty_cache()

    model.eval()

    if get_rank() == 0:
        pbar = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True)
    else:
        pbar = enumerate(loader)

    preds = {}

    meshes, _ = load_bop_meshes(cfg['DATASETS']['MESH_DIR'])

    for idx, (images, targets, meta_infos) in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        pred, aux = model(images, targets=targets)

        if get_rank() == 0 and idx % 10 == 0:
            bIdx = 0
            imgpath, imgname = os.path.split(meta_infos[bIdx]['path'])
            name_prefix = imgpath.replace(os.sep, '_').replace('.', '') + '_' + os.path.splitext(imgname)[0]

            rawImg, visImg, gtImg = visualize_pred(images.tensors[bIdx], targets[bIdx], pred[bIdx], 
                cfg['INPUT']['PIXEL_MEAN'],  cfg['INPUT']['PIXEL_STD'], meshes)
            # cv2.imwrite(cfg['RUNTIME']['WORKING_DIR'] + name_prefix + '.png', rawImg)
            cv2.imwrite(cfg['RUNTIME']['WORKING_DIR'] + name_prefix + '_pred.png', visImg)
            cv2.imwrite(cfg['RUNTIME']['WORKING_DIR'] + name_prefix + '_gt.png', gtImg)

        # pred = [p.to('cpu') for p in pred]
        for m, p in zip(meta_infos, pred):
            preds.update({m['path']:{
                'meta': m,
                'pred': p
            }})

    preds = accumulate_dicts(preds)

    if get_rank() != 0:
        return
    
    accuracy_adi_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range \
        = evaluate(cfg, preds)

    print_accuracy_per_class(accuracy_adi_per_class, accuracy_rep_per_class)

    # writing log to tensorboard
    if logger:
        classNum = cfg['DATASETS']['N_CLASS'] - 1 # get rid of background class        
        assert(len(accuracy_adi_per_class) == classNum)
        assert(len(accuracy_rep_per_class) == classNum)

        all_adi = {}
        all_rep = {}
        validClassNum = 0

        for i in range(classNum):
            className = ('class_%02d' % i)
            logger.add_scalars('ADI/' + className, accuracy_adi_per_class[i], epoch)
            logger.add_scalars('REP/' + className, accuracy_rep_per_class[i], epoch)
            # 
            assert(len(accuracy_adi_per_class[i]) == len(accuracy_rep_per_class[i]))
            if len(accuracy_adi_per_class[i]) > 0:
                for key, val in accuracy_adi_per_class[i].items():
                    if key in all_adi:
                        all_adi[key] += val
                    else:
                        all_adi[key] = val
                for key, val in accuracy_rep_per_class[i].items():
                    if key in all_rep:
                        all_rep[key] += val
                    else:
                        all_rep[key] = val
                validClassNum += 1

        # averaging
        for key, val in all_adi.items():
            all_adi[key] = val / validClassNum
        for key, val in all_rep.items():
            all_rep[key] = val / validClassNum  
        logger.add_scalars('ADI/all_class', all_adi, epoch)
        logger.add_scalars('REP/all_class', all_rep, epoch)

    return accuracy_adi_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range

def train(cfg, epoch, max_epoch, loader, model, optimizer, scheduler, device, logger=None):
    model.train()

    if get_rank() == 0:
        pbar = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True)
    else:
        pbar = enumerate(loader)

    for idx, (images, targets, _) in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        _, loss_dict = model(images, targets=targets)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_reg = loss_dict['loss_reg'].mean()

        loss = loss_cls + loss_reg
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        loss_reduced = reduce_loss_dict(loss_dict)
        loss_cls = loss_reduced['loss_cls'].mean().item()
        loss_reg = loss_reduced['loss_reg'].mean().item()

        if get_rank() == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar_str = (("epoch: %d/%d, lr:%.6f, cls:%.4f, reg:%.4f") % (epoch+1, max_epoch, current_lr, loss_cls, loss_reg))
            pbar.set_description(pbar_str)

            # writing log to tensorboard
            if logger and idx % 10 == 0:
                # totalStep = (epoch * len(loader) + idx) * args.batch * args.n_gpu
                totalStep = (epoch * len(loader) + idx) * cfg['SOLVER']['IMS_PER_BATCH']
                logger.add_scalar('training/learning_rate', current_lr, totalStep)
                logger.add_scalar('training/loss_cls', loss_cls, totalStep)
                logger.add_scalar('training/loss_reg', loss_reg, totalStep)
                logger.add_scalar('training/loss_all', (loss_cls + loss_reg), totalStep)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return sampler.RandomSampler(dataset)

    else:
        return sampler.SequentialSampler(dataset)


if __name__ == '__main__':
    cfg = get_args()

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

    train_trans = transform.Compose(
        [
            transform.Resize(
                cfg['INPUT']['INTERNAL_WIDTH'], 
                cfg['INPUT']['INTERNAL_HEIGHT'], internal_K),
            transform.RandomShiftScaleRotate(
                cfg['SOLVER']['AUGMENTATION_SHIFT'], 
                cfg['SOLVER']['AUGMENTATION_SCALE'], 
                cfg['SOLVER']['AUGMENTATION_ROTATION'], 
                cfg['INPUT']['INTERNAL_WIDTH'], 
                cfg['INPUT']['INTERNAL_HEIGHT'], 
                internal_K),
            transform.Normalize(
                cfg['INPUT']['PIXEL_MEAN'], 
                cfg['INPUT']['PIXEL_STD']),
            transform.ToTensor(),
        ]
    )

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

    train_set = BOP_Dataset(
        cfg['DATASETS']['TRAIN'], 
        cfg['DATASETS']['MESH_DIR'], 
        cfg['DATASETS']['BBOX_FILE'], 
        train_trans,
        cfg['SOLVER']['STEPS_PER_EPOCH'] * cfg['SOLVER']['IMS_PER_BATCH'],
        training = True)
    valid_set = BOP_Dataset(
        cfg['DATASETS']['VALID'],
        cfg['DATASETS']['MESH_DIR'], 
        cfg['DATASETS']['BBOX_FILE'], 
        valid_trans,
        training = False)

    if cfg['MODEL']['BACKBONE'] == 'darknet53':
        backbone = darknet53(pretrained=True)
    else:
        print("unsupported backbone!")
        assert(0)
    model = PoseModule(cfg, backbone)
    model = model.to(device)
    start_epoch = 0

    # https://discuss.pytorch.org/t/is-average-the-correct-way-for-the-gradient-in-distributeddataparallel-with-multi-nodes/34260/13
    base_lr = cfg['SOLVER']['BASE_LR'] / cfg['RUNTIME']['N_GPU']
    optimizer = optim.SGD(
        model.parameters(),
        lr = 0, # the learning rate will be taken care by scheduler
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )

    batch_size_per_gpu = int(cfg['SOLVER']['IMS_PER_BATCH'] / cfg['RUNTIME']['N_GPU'])
    max_epoch = math.ceil(cfg['SOLVER']['MAX_ITER'] * cfg['SOLVER']['IMS_PER_BATCH'] / len(train_set))

    scheduler_batch = WarmupScheduler(
        optimizer, base_lr, 
        cfg['SOLVER']['MAX_ITER'], cfg['SOLVER']['SCHEDULER_POLICY'], cfg['SOLVER']['SCHEDULER_PARAMS'])

    if cfg['RUNTIME']['DISTRIBUTED']:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg['RUNTIME']['LOCAL_RANK']],
            output_device=cfg['RUNTIME']['LOCAL_RANK'],
            broadcast_buffers=False,
        )
        model = model.module

    # load weight and create working_dir dynamically
    timestr = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
    name_wo_ext = os.path.splitext(os.path.split(cfg['RUNTIME']['CONFIG_FILE'])[1])[0]
    working_dir = 'working_dirs' + '/' + name_wo_ext + '/' + timestr + '/'
    cfg['RUNTIME']['WORKING_DIR'] = working_dir
    if os.path.exists(cfg['RUNTIME']['WEIGHT_FILE']):
        try:
            chkpt = torch.load(cfg['RUNTIME']['WEIGHT_FILE'], map_location='cpu')  # load checkpoint
            if 'model' in chkpt:
                assert('steps' in chkpt and 'optim' in chkpt)
                scheduler_batch.step_multiple(chkpt['steps'])
                start_epoch = int(chkpt['steps'] * cfg['SOLVER']['IMS_PER_BATCH'] / len(train_set))
                model.load_state_dict(chkpt['model'])
                optimizer.load_state_dict(chkpt['optim'])
                # update working dir
                cfg['RUNTIME']['WORKING_DIR'] = os.path.split(cfg['RUNTIME']['WEIGHT_FILE'])[0] + '/'
                print('Weights and optimzer are loaded from ' + cfg['RUNTIME']['WEIGHT_FILE'])
            else:
                model.load_state_dict(chkpt)
                print('Weights from are loaded from ' + cfg['RUNTIME']['WEIGHT_FILE'])
        except:
            pass
    else:
        pass
    # 
    print("working directory: " + cfg['RUNTIME']['WORKING_DIR'])
    if get_rank() == 0:
        os.makedirs(cfg['RUNTIME']['WORKING_DIR'], exist_ok=True)
        logger = SummaryWriter(cfg['RUNTIME']['WORKING_DIR'])

    # compute model size
    total_params_count = sum(p.numel() for p in model.parameters())
    print("Model size: %d parameters" % total_params_count)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size_per_gpu,
        sampler=data_sampler(train_set, shuffle=True, distributed=cfg['RUNTIME']['DISTRIBUTED']),
        num_workers=cfg['RUNTIME']['NUM_WORKERS'],
        collate_fn=collate_fn(cfg['INPUT']['SIZE_DIVISIBLE']),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size_per_gpu,
        sampler=data_sampler(valid_set, shuffle=False, distributed=cfg['RUNTIME']['DISTRIBUTED']),
        num_workers=cfg['RUNTIME']['NUM_WORKERS'],
        collate_fn=collate_fn(cfg['INPUT']['SIZE_DIVISIBLE']),
    )

    # write cfg to working_dir
    with open(cfg['RUNTIME']['WORKING_DIR'] + 'cfg.json', 'w') as f:
        json.dump(cfg, f, indent=4, sort_keys=True)

    for epoch in range(start_epoch, max_epoch):
        train(cfg, epoch, max_epoch, train_loader, model, optimizer, scheduler_batch, device, logger=logger)

        valid(cfg, epoch, valid_loader, model, device, logger=logger)

        if get_rank() == 0:
            torch.save({
                'steps': (epoch + 1) * int(len(train_set) / cfg['SOLVER']['IMS_PER_BATCH']), 
                'model': model.state_dict(), 
                'optim': optimizer.state_dict(),
                },
                cfg['RUNTIME']['WORKING_DIR'] + 'latest.pth',
            )
            if epoch == (max_epoch - 1):
                torch.save(model.state_dict(), cfg['RUNTIME']['WORKING_DIR'] + 'final.pth')

    # output final info
    if get_rank() == 0:
        timestr = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
        commandstr = ' '.join([str(elem) for elem in sys.argv]) 
        final_msg = ("finished at: %s\nworking_dir: %s\ncommands:%s" % (timestr, cfg['RUNTIME']['WORKING_DIR'], commandstr))
        with open(cfg['RUNTIME']['WORKING_DIR'] + 'info.txt', 'w') as f:
            f.write(final_msg)
        print(final_msg)
