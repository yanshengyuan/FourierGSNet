import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
#from utils.loader import get_training_data,get_validation_data
from utils.loader import  get_training_data,get_validation_data
from model_ICLR import GSNet_Simu as PImodel

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Define arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--loss', default='mse', type=str,
                    help='Loss function to be optimized during training - "mse" for MeanSquaredError, "mae" for MeanAbsoluteError')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='Optimizer algorithm - "adam" for Adam optimizer, "sgd" for SGD optimizer.')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--step_size', default=30, type=int,
                    help='step size (default: 30)')
parser.add_argument('--gamma', default=0.5, type=int,
                    help='gamma (default: 0.5)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-freq', '--print_freq', default=1, type=int)
parser.add_argument('--pth_name', default='tmp.pth.tar', type=str)
parser.add_argument('--gridsize', default=256*1e-6, type=float)
parser.add_argument('--wavelength', default=1e-10, type=float)
parser.add_argument('--prop_dist', default=0.1, type=float)
parser.add_argument('--val_vis_path', type=str)
parser.add_argument('--log_features', default=False, type=bool)

# Type checking
args = parser.parse_args()
if not isinstance(args.data, str):
    raise TypeError('data must be an instance of str!')#the path to the root directory of training dataset
if not isinstance(args.workers, int):
    raise TypeError('workers must be an instance of int!')#number of threads opened for fetching data from drive to GPU
if not isinstance(args.epochs, int):
    raise TypeError('epochs path must be an instance of int!')#recommend training for 1000 epochs because the best model on validation dataset often appeas around 700th epo
if not isinstance(args.start_epoch, int):
    raise TypeError('start_epoch must be an instance of int!')#start epoch is usually set as 0 unless you resume training from a checkpoint
if not isinstance(args.batch_size, int):
    raise TypeError('batch_size must be an instance of int!')#batchsize as 128 is recommended for 10k training data cooped with initial learning rate as 5e-3 for resnet18 and 1e-3 for other models
if not isinstance(args.loss, str):
    raise TypeError('loss must be an instance of str!')#MSE loss is recommended rather than MAE because MAE's derivative is a constant value
if not isinstance(args.optimizer, str):
    raise TypeError('optimizer must be an instance of str!')#Adam or SGD
if not isinstance(args.lr, float):
    raise TypeError('lr must be an instance of float!')#the initial learning rate for Adam or SGD
if not isinstance(args.momentum, float):
    raise TypeError('momentum must be an instance of float!')#momentum is an important parameter for SGD optimizor
if not isinstance(args.weight_decay, float):
    raise TypeError('weight_decay must be an instance of float!')#weight decay is recommended to be fixed as 1e-3 for optimizors
if not isinstance(args.step_size, int):
    raise TypeError('step_size must be an instance of int!')#stepsize is the length between every time you want the lr to decay in lr schedulers. 30 is the recommended value for 1000 epochs
if not isinstance(args.gamma, float):
    raise TypeError('gamma must be an instance of float!')#gamma is the ratio every time you want the lr to decay by in the lr schedulers. 0.5 is the recommended value for 1000 epochs
if not isinstance(args.print_freq, int):
    raise TypeError('print_freq must be an instance of int!')#the frequency to print updated loss and batch time on screen
if not isinstance(args.resume, str):
    raise TypeError('resume must be an instance of str!')#when you want to resume training from a checkpoint you need to give a path of checkpoint in this parameter
if not isinstance(args.evaluate, bool):
    raise TypeError('evaluate must be an instance of bool!')#to specify whether you are training or evaluating. This will determine to mode of the model to be train or eval
if not isinstance(args.world_size, int):
    raise TypeError('world_size must be an instance of int!')#the number of machines when you are doing a distributed parallel training
if not isinstance(args.rank, int):
    raise TypeError('rank must be an instance of int!')#the rank of priorities for different machines to use when doing a distributed parallel training
if not isinstance(args.dist_url, str):
    raise TypeError('dist_url must be an instance of str!')#url(IP) for machines when doing a distributed parallel training
if not isinstance(args.dist_backend, str):
    raise TypeError('dist_backend must be an instance of str!')#backends for a distributed parallel training


best_acc1 = float('inf')

def main():
    timestart = time.perf_counter()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    timeend=time.perf_counter()
    print("Total Time: %d"%(timeend-timestart))



def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    traindir = os.path.join(args.data)
    valdir = os.path.join(args.data)

    train_dataset = get_training_data(traindir)
    val_dataset = get_validation_data(valdir)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    #Determine the size of the initial phase and source intensity
    H=0
    W=0
    for i, (I, Phi) in enumerate(train_loader):
        B, C, H, W = I.shape
        print(H, W)
        break
    if(H!=W):
        print("warning! Initial field must be in square shape!")
    if torch.cuda.is_available():
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
    
    gen_phase_init = (torch.rand((args.batch_size, H, W), device='cuda'))*2*torch.pi-torch.pi
    gs_phase_init = (torch.rand((args.batch_size, H, W), device='cuda'))*2*torch.pi-torch.pi
    model = PImodel(args)
    print(model)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # define loss function (criterion)
    if args.loss == 'mse':
        criterion = nn.MSELoss().to(device)
    elif args.loss == 'mae':
        criterion = nn.L1Loss().to(device)
    else: ValueError('The loss function is not valid!')

    # define learning rate
    if args.lr > 0:
        lr = args.lr
    else: ValueError('Learning rate must be greater than 0!')

    # define momentum
    if args.momentum > 0:
        momentum = args.momentum
    else: ValueError('Momentum must be greater than 0!')

    # define weight decay
    if args.weight_decay > 0:
        weight_decay = args.weight_decay
    else: ValueError('Weight decay must be greater than 0!')

    # define optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)
    else: ValueError('The optimizer is not valid!')

    # define step size
    if args.step_size > 0:
        step_size = args.step_size
    else: ValueError('Step size must be greater than zero!')

    # define gamma
    if args.gamma > 0:
        gamma = args.gamma
    else: ValueError('Gamma must be greater than zero!')

    """Sets the learning rate to the initial LR decayed by specified number of epochs"""
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    if args.evaluate:
        checkpoint_name = args.pth_name
        print("=> loading checkpoint '{}'".format(checkpoint_name))
        if args.gpu is None:
            checkpoint = torch.load(checkpoint_name)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(checkpoint_name, map_location=loc)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_name, checkpoint['epoch']))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        else:
            train_sampler = None
            val_sampler = None

        acc2 = validate(val_loader, model, criterion, args, checkpoint['epoch'], args.val_vis_path)
        return

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, epoch, args.val_vis_path)
        
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 < best_acc1
        best_acc1 = min(acc1, best_acc1)
        print("Best: %f"%best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, args.pth_name)
        
        '''
        if(args.log_features==True):
            regular_pth_path = os.path.join( args.val_vis_path, str(epoch)+"pth.tar" )
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, True, regular_pth_path)
        '''


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (I, Phi) in enumerate(train_loader):
        #if(i==1): break;
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        I_far = I.to(device, non_blocking=True).squeeze()
        Phi=Phi.squeeze()
        Phi = Phi.to(device, non_blocking=True)

        # compute output
        output = model(I_far).squeeze()
        loss = criterion(output, Phi)

        # measure accuracy and record loss
        losses.update(loss.item(), Phi.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(optimizer.param_groups[0]['lr'])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args, epoch, output_dir):
    def run_validate(loader, epoch, output_dir, base_progress=0):
        avg=0
        with torch.no_grad():
            end = time.time()
            num_batch=0
            for x, (I, Phi) in enumerate(loader):
                #if(num_batch==1): break;
                x = base_progress + x
                if args.gpu is not None and torch.cuda.is_available():
                    I_far = I.cuda(args.gpu, non_blocking=True).squeeze()

                Phi=Phi.squeeze(0)
                if torch.backends.mps.is_available():
                    I = I.to('mps')
                    Phi = Phi.to('mps')
                if torch.cuda.is_available():
                    Phi = Phi.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(I_far).squeeze(0)
                num_batch += 1
                for i in range(len(output)):
                    outdata=output[i].squeeze()
                    gtdata=Phi[i].squeeze().cpu()
                    outdata=outdata.cpu()
                    np.save(output_dir+'/Phi_pred/npy/'+str(x)+'_'+str(i)+'.npy', outdata)
                    plt.imsave(output_dir+'/Phi_pred/img/'+str(x)+'_'+str(i)+'.png' ,outdata, cmap='gray')
                    np.save(output_dir+'/Phi_gt/npy/'+str(x)+'_'+str(i)+'.npy', gtdata)
                    plt.imsave(output_dir+'/Phi_gt/img/'+str(x)+'_'+str(i)+'.png' ,gtdata, cmap='gray')
                    '''
                    GTdata=I_far[i].squeeze().cpu()
                    np.save(output_dir+'/I_gt/npy/'+str(x)+'_'+str(i)+'.npy', GTdata)
                    plt.imsave(output_dir+'/I_gt/img/'+str(x)+'_'+str(i)+'.png' , GTdata, cmap='gray')
                    '''
                    
                
                loss = criterion(output, Phi)
                avg+=loss.item()

                # measure accuracy and record loss
                losses.update(loss.item(), Phi.size(0))
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                #if x % args.print_freq == 0:
                progress.display(x + 1)
        return avg

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    l=run_validate(val_loader, epoch, output_dir)/len(val_loader)

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, epoch, output_dir, len(val_loader))

    progress.display_summary()

    return l


def save_checkpoint(state, is_best, model_name):
    if is_best:
        torch.save(state, model_name)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
