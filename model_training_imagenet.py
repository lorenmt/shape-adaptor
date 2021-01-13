import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torchvision.datasets as dset
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from torch.hub import load_state_dict_from_url
from thop import profile
from thop import clever_format
import json
import pdb

from model_list import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data', metavar='DIR', help='path to dataset')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

### shape adaptor related flags
parser.add_argument('--mode', default='shape-adaptor', type=str, help='human-cifar, human-imagenet, shape-adaptor, autosc')
parser.add_argument('--network', default='vgg', type=str, help='vgg, resnet, mobilenetv2')
parser.add_argument('--step', default=1500, type=int, help='the step between two shape adaptor updates, 200 for autosc, 1500 for standard')
parser.add_argument('--output_dim', default=8, type=int, help='output dim required for shape adaptor initialisations')
parser.add_argument('--limit_dim', default=15, type=int, help='limit dim required for memory constraint shape adaptor')
parser.add_argument('--sa_num', default=None, type=int, help='the number of shape adaptor, use None to compute automatically')
parser.add_argument('--width_mult', default=1.0, type=float, help='width multiplier only for MobileNetv2')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

best_acc = 0

def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
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

    ngpus_per_node = torch.cuda.device_count()
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


def main_worker(gpu, ngpus_per_node, args):
    global best_acc
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

    # create model
    if args.network == 'vgg':
        model = VGG(dataset='imagenet', input_shape=224, output_shape=args.output_dim, mode=args.mode, type='D')
    if args.network == 'resnet':
        model = ResNet(Bottleneck, [3, 4, 6, 3], dataset='imagenet', mode=args.mode,
                       input_shape=224, output_shape=args.output_dim)
    elif args.network == 'mobilenetv2':
        model = MobileNetV2(sa_num=args.sa_num, dataset='imagenet', mode=args.mode,
                            input_shape=224, output_shape=args.output_dim, width_mult=args.width_mult)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        model = torch.nn.DataParallel(model).cuda()
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.network.startswith('alexnet') or args.network.startswith('vgg'):
            #  model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define logging function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    alpha_list = []
    parameter_list = []
    for _, (key, value) in enumerate(model.named_parameters()):
        if 'alpha' in key:
            alpha_list.append(value)
        else:
            parameter_list.append(value)

    alpha_optimizer = None
    if 'human' not in args.mode:
        alpha_optimizer = optim.SGD(alpha_list, lr=0.1, momentum=0.9, nesterov=True)
        alpha_scheduler = optim.lr_scheduler.CosineAnnealingLR(alpha_optimizer, args.epochs)

    if args.network == 'mobilenetv2':
        weight_optimizer = optim.SGD(parameter_list, lr=0.05, weight_decay=4e-5, momentum=0.9, nesterov=True)
    else:
        weight_optimizer = optim.SGD(parameter_list, lr=0.1, weight_decay=1e-4, momentum=0.9, nesterov=True)

    weight_scheduler = optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, args.epochs)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    trans_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    trans_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # im_root = '/trainman-mount/trainman-storage-aae20046-12f7-43f6-9bff-91d953a173a5/ImageNet'
    im_root = args.data
    train_set = dset.ImageNet(root=im_root, split='train', transform=trans_train)
    val_set = dset.ImageNet(root=im_root, split='val', transform=trans_test)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        tic = time.time()
        loss_val, train_acc = weight_train(train_loader, model, criterion, weight_optimizer, alpha_optimizer, epoch, args)
        toc = time.time()
        print("training weight takes {}".format(toc - tic))

        test_acc1, test_acc5 = validate(val_loader, model, criterion, args)
        input_data = torch.randn(1, 3, 224, 224).cuda()
        flops, params = profile(model, inputs=(input_data, ), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        if 'human' not in args.mode:
            current_state = 'EPOCH {:03d}|TRAIN_ACC {}|TEST_TOP1 {}|TEST_TOP5 {}|SCALING {}|SHAPE {}|Macs {}|Para {}'\
                .format(epoch, train_acc.data, test_acc1.data, test_acc5.data, ShapeAdaptor.penalty, model.module.shape_list, flops, params)
        else:
            current_state = 'EPOCH {:03d}|TRAIN_ACC {}|TEST_TOP1 {}|TEST_TOP5 {}|Macs {}|Para {}' \
                .format(epoch, train_acc.data, test_acc1.data, test_acc5.data, flops, params)

        path_name = '{}-{}-step{}-output_dim{}'.format(args.network, args.mode, args.step, args.output_dim)
        f = open('logging/{}.csv'.format(path_name), 'a')
        f.write('{}\n'.format(current_state))
        f.close()

        is_best = test_acc1 > best_acc
        best_acc = max(test_acc1, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'acc': test_acc1,
            'best_acc': best_acc,
        }, is_best, checkpoint='logging', filename=path_name)

        weight_scheduler.step()
        if 'human' not in args.mode:
            alpha_scheduler.step()
            print(model.module.shape_list)


def weight_train(train_loader, model, criterion, weight_optimizer, alpha_optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute gradient and do SGD step
        weight_optimizer.zero_grad()
        if 'human' not in args.mode:
            alpha_optimizer.zero_grad()

        # update alphas
        if i % args.step == 0 and 'human' not in args.mode:
            # compute output
            if (i == 0 and epoch == 0) or ShapeAdaptor.current_dim_true < args.limit_dim:
                ShapeAdaptor.penalty = 1.0
            else:
                scaling_total = args.limit_dim / ShapeAdaptor.current_dim_true
                ShapeAdaptor.penalty = np.power(scaling_total, 1 / ShapeAdaptor.counter)

            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            alpha_optimizer.step()

            weight_optimizer.zero_grad()
            alpha_optimizer.zero_grad()

        # update weight parameters
        if 'human' not in args.mode:
            if (i == 0 and epoch == 0) or ShapeAdaptor.current_dim_true < args.limit_dim:
                ShapeAdaptor.penalty = 1.0
            else:
                scaling_total = args.limit_dim / ShapeAdaptor.current_dim_true
                ShapeAdaptor.penalty = np.power(scaling_total, 1 / ShapeAdaptor.counter)

        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        weight_optimizer.step()

        # measure accuracy and record logging
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed timse
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if 'human' not in args.mode:
                if ShapeAdaptor.current_dim_true < args.limit_dim:
                    ShapeAdaptor.penalty = 1.0
                else:
                    scaling_total = args.limit_dim / ShapeAdaptor.current_dim_true
                    ShapeAdaptor.penalty = np.power(scaling_total, 1 / ShapeAdaptor.counter)

            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record logging
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
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

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint'):
    filepath = os.path.join(checkpoint, filename+'.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best_' + filename + '.pth'))


if __name__ == '__main__':
    main()
