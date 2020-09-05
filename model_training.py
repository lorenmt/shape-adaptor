import torch
import torch.nn as nn
import numpy as np

from thop import profile
from thop import clever_format
from model_list import *
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] ='True'
parser = argparse.ArgumentParser(description='Shape Adaptor standard and AutoSC mode training for single GPU')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10, cifar100, svhn, aircraft, stanford-cars, cub_200')
parser.add_argument('--mode', default='shape-adaptor', type=str, help='human-cifar, human-imagenet, shape-adaptor, autosc')
parser.add_argument('--network', default='vgg', type=str, help='vgg, resnet, mobilenetv2')
parser.add_argument('--sa_num', default=None, type=int, help='the number of shape adaptor, use None to compute automatically')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate for network weights, 0.1 for small,  0.01 for large')
parser.add_argument('--id', default=0, type=int, help='for statistical evaluation purpose')
parser.add_argument('--input_dim', default=32, type=int, help='32 for small, 224 for large')
parser.add_argument('--output_dim', default=8, type=int, help='output dim required for shape adaptor initialisations')
parser.add_argument('--limit_dim', default=999, type=int, help='limit dim required for memory constraint shape adaptor')
parser.add_argument('--step', default=20, type=int, help='step between every alpha parameters updates')
parser.add_argument('--batch_size', default=128, type=int, help='128 for small, 8 for large')
parser.add_argument('--epochs', default=200, type=int, help='total training epochs')
parser.add_argument('--width_mult', default=1.0, type=float, help='width multiplier only for MobileNetv2')

args = parser.parse_args()


def model_fit(x_pred, x_output):
    loss = F.cross_entropy(x_pred, x_output)
    return loss


# define image transformation
batch_size = args.batch_size
trans_train = transforms.Compose([
    transforms.Resize(256) if args.input_dim == 224 else transforms.Resize(32),
    transforms.RandomCrop(224) if args.input_dim == 224 else transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN[args.dataset], VAR[args.dataset]),
])

trans_test = transforms.Compose([
    transforms.Resize(256) if args.input_dim == 224 else transforms.Resize(32),
    transforms.CenterCrop(args.input_dim),
    transforms.ToTensor(),
    transforms.Normalize(MEAN[args.dataset], VAR[args.dataset]),
])

# use official pytorch datasets if available
if args.dataset == 'cifar10':
    train_set = dset.CIFAR10(root='dataset', train=True, transform=trans_train, download=True)
    test_set  = dset.CIFAR10(root='dataset', train=False, transform=trans_test, download=True)

elif args.dataset == 'cifar100':
    train_set = dset.CIFAR100(root='dataset', train=True, transform=trans_train, download=True)
    test_set  = dset.CIFAR100(root='dataset', train=False, transform=trans_test, download=True)

elif args.dataset == 'svhn':
    train_set = dset.SVHN(root='dataset', split='train', transform=trans_train, download=True)
    test_set  = dset.SVHN(root='dataset', split='test', transform=trans_test, download=True)


if args.dataset in ['cifar10', 'cifar100', 'svhn']:
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)
else:
    train_loader = torch.utils.data.DataLoader(
        dataset=dset.ImageFolder('dataset/{:s}/train'.format(args.dataset), transform=trans_train),
        batch_size=batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=dset.ImageFolder('dataset/{:s}/test'.format(args.dataset), transform=trans_test),
        batch_size=batch_size,
        shuffle=False)


# define network
device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")
if args.network == 'vgg':
    model = VGG(type='D', sa_num=args.sa_num,
                dataset=args.dataset, mode=args.mode,
                input_shape=args.input_dim, output_shape=args.output_dim).to(device)
elif args.network == 'resnet':
    model = ResNet(Bottleneck, [3, 4, 6, 3], sa_num=args.sa_num,
                   dataset=args.dataset, mode=args.mode,
                   input_shape=args.input_dim, output_shape=args.output_dim).to(device)
elif args.network == 'mobilenetv2':
    model = MobileNetV2(sa_num=args.sa_num, dataset=args.dataset, mode=args.mode,
                        input_shape=args.input_dim, output_shape=args.output_dim, width_mult=args.width_mult).to(device)


# define individual parameter lists for network shape and network weight
alpha_list = []
weight_list = []
for _, (key, value) in enumerate(model.named_parameters()):
    if 'alpha' in key:
        alpha_list.append(value)
    else:
        weight_list.append(value)

total_epoch = args.epochs
if 'human' not in args.mode:
    alpha_optimizer = optim.SGD(alpha_list, lr=0.1, momentum=0.9, nesterov=True)
    alpha_scheduler = optim.lr_scheduler.CosineAnnealingLR(alpha_optimizer, total_epoch)

if args.network == 'mobilenetv2':
    # this learning rate is suggested in the official MobileNetv2 paper
    weight_optimizer = optim.SGD(weight_list, lr=args.lr, weight_decay=4e-5, momentum=0.9, nesterov=True)
else:
    weight_optimizer = optim.SGD(weight_list, lr=args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
weight_scheduler = optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, total_epoch)


train_batch = len(train_loader)
test_batch = len(test_loader)
iteration = 0
avg_cost = np.zeros([total_epoch, 4], dtype=np.float32)
shape_list = []
sigmoid_alpha_list = []
penalty_list = []
start_time = time.time()  # start processing time

for index in range(total_epoch):
    cost = np.zeros(2, dtype=np.float32)

    # evaluate training data
    model.train()
    train_dataset = iter(train_loader)
    for i in range(train_batch):
        train_data, train_label = train_dataset.next()
        train_label = train_label.type(torch.LongTensor)
        train_data, train_label = train_data.to(device), train_label.to(device)

        # update alpha
        if 'human' not in args.mode and i % args.step == 0:
            if iteration == 0 or ShapeAdaptor.current_dim_true < args.limit_dim:
                ShapeAdaptor.penalty = 1.0
            else:
                # compute penalty for alpha here
                penalty_total = args.limit_dim / ShapeAdaptor.current_dim_true
                ShapeAdaptor.penalty = np.power(penalty_total, 1 / ShapeAdaptor.counter)

            train_pred = model(train_data)
            train_loss = model_fit(train_pred, train_label)
            train_loss.backward()
            alpha_optimizer.step()

            weight_optimizer.zero_grad()
            alpha_optimizer.zero_grad()

        # update weight with updated alphas
        if 'human' not in args.mode:
            if iteration == 0 or ShapeAdaptor.current_dim_true < args.limit_dim:
                ShapeAdaptor.penalty = 1.0
            else:
                penalty_total = args.limit_dim / ShapeAdaptor.current_dim_true
                ShapeAdaptor.penalty = np.power(penalty_total, 1 / ShapeAdaptor.counter)

        train_pred = model(train_data)
        train_loss = model_fit(train_pred, train_label)
        train_loss.backward()
        weight_optimizer.step()

        weight_optimizer.zero_grad()
        if 'human' not in args.mode:
            alpha_optimizer.zero_grad()

        # compute training data accuracy
        train_predict_label1 = train_pred.data.max(1)[1]
        train_acc1 = train_predict_label1.eq(train_label).sum().item() / train_data.shape[0]

        cost[0] = torch.mean(train_loss).item()
        cost[1] = train_acc1
        iteration += 1
        avg_cost[index][0:2] += cost / train_batch

    # evaluating test data
    model.eval()
    with torch.no_grad():
        test_dataset = iter(test_loader)
        for i in range(test_batch):
            test_data, test_label = test_dataset.next()
            test_label = test_label.type(torch.LongTensor)
            test_data, test_label = test_data.to(device), test_label.to(device)

            if args.mode == 'shape-adaptor':
                if ShapeAdaptor.current_dim_true < args.limit_dim:
                    ShapeAdaptor.penalty = 1.0
                else:
                    penalty_total = args.limit_dim / ShapeAdaptor.current_dim_true
                    ShapeAdaptor.penalty = np.power(penalty_total, 1 / ShapeAdaptor.counter)

            test_pred = model(test_data)
            test_loss = model_fit(test_pred, test_label)

            # compute test data accuracy
            test_predict_label1 = test_pred.data.max(1)[1]
            test_acc1 = test_predict_label1.eq(test_label).sum().item() / test_data.shape[0]

            cost[0] = torch.mean(test_loss).item()
            cost[1] = test_acc1
            avg_cost[index][2:] += cost / test_batch

    # scheduler update
    weight_scheduler.step()
    if 'human' not in args.mode:
        alpha_scheduler.step()

    # compute memory and parameter usage
    input_data = torch.randn(1, 3, args.input_dim, args.input_dim).to(device)
    flops, params = profile(model, inputs=(input_data, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print('EPOCH: {:04d} ITER: {:04d} | TRAIN [LOSS|ACC.]: {:.4f} {:.4f} || TEST [LOSS|ACC.]: {:.4f} {:.4f} || MACs {} Params {}'
          .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2], avg_cost[index][3], flops, params))

    if 'human' not in args.mode:
        alphas = [0.5 + 0.5 * torch.sigmoid(i).squeeze().detach().cpu().numpy() for i in alpha_list]
        sigmoid_alpha_list.append(alphas)
        shape_list.append(model.shape_list)
        penalty_list.append(ShapeAdaptor.penalty)
        print('s(alpha) = {} | current shape = {}'.format(sigmoid_alpha_list[-1], shape_list[-1]))
    else:
        shape_list.append(model.shape_list)
        print('human designed shape = {}'.format(shape_list[-1]))
    print('TOP: {}'.format(max(avg_cost[:, 3])))

end_time = time.time()
print('Total training takes {:.4f} seconds.'.format(end_time - start_time))

if 'human' in args.mode:
    dict = {'loss': avg_cost,
            'macs': flops,
            'shape-list': shape_list,
            'parms': params,
            'time': end_time - start_time}
    np.save('logging/{}_{}_{}_{}.npy'
            .format(args.dataset, args.network, args.mode, args.id), dict)
else:
    dict = {'shape-list': shape_list,
            'alpha': sigmoid_alpha_list,
            'penalty': penalty_list,
            'loss': avg_cost,
            'macs': flops,
            'parms': params,
            'time': end_time - start_time}
    np.save('logging/{}_{}_{}_sanum{}_outputdim{}_{}.npy'
            .format(args.dataset, args.network, args.mode, len(shape_list[0]) - 1, args.output_dim, args.id), dict)

