import torch
import torch.nn as nn
import numpy as np

from torch.hub import load_state_dict_from_url
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
parser = argparse.ArgumentParser(description='AutoTL training for single GPU')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--dataset', default='cub_200', type=str, help='cub_200, wikiart, sketches, stanford-cars, vgg-flowers')
parser.add_argument('--alpha_lr', default=0.1, type=float, help='learning rate for shape parameters')
parser.add_argument('--weight_lr', default=0.001, type=float, help='learning rate for weight parameters')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for network weight')
parser.add_argument('--id', default=0, type=int, help='for statistical evaluation purpose')
parser.add_argument('--limit_dim', default=999, type=int, help='limit dim for memory constraint version')
parser.add_argument('--step', default=20, type=int, help='the step between two shape adaptor updates')
parser.add_argument('--batch-size', default=8, type=int, help='batch size')
parser.add_argument('--epochs', default=80, type=int, help='total training epochs')

args = parser.parse_args()


def model_fit(x_pred, x_output):
    loss = F.cross_entropy(x_pred, x_output)
    return loss


# define image transformation
batch_size = args.batch_size
if args.dataset in ['stanford-cars', 'cub_200']:
    # we apply the same transformation in:
    # https://github.com/arunmallya/piggyback/blob/5a6094c45896c035a690d6f2fac0b102df176600/src/dataset.py
    trans_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN[args.dataset], VAR[args.dataset]),
    ])

    trans_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN[args.dataset], VAR[args.dataset]),
    ])
else:
    trans_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN[args.dataset], VAR[args.dataset]),
    ])

    trans_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN[args.dataset], VAR[args.dataset]),
    ])

train_loader = torch.utils.data.DataLoader(
    dataset=dset.ImageFolder('dataset/autotl/{:s}/train'.format(args.dataset), transform=trans_train),
    batch_size=batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=dset.ImageFolder('dataset/autotl/{:s}/test'.format(args.dataset), transform=trans_test),
    batch_size=batch_size,
    shuffle=False)

# define a pre-trained ResNet-50 network
device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")

model = ResNet(Bottleneck, [3, 4, 6, 3], dataset=args.dataset, mode='autotl', input_shape=224).to(device)
model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
state_dict = load_state_dict_from_url(model_url)
state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
model.load_state_dict(state_dict, strict=False)


# define individual parameter lists for network shape and network weight
alpha_list = []
weight_list = []
for _, (key, value) in enumerate(model.named_parameters()):
    if 'alpha' in key:
        alpha_list.append(value)
    else:
        weight_list.append(value)


total_epoch = args.epochs
alpha_optimizer = optim.SGD(alpha_list, lr=args.alpha_lr, momentum=0.9, nesterov=True)
alpha_scheduler = optim.lr_scheduler.CosineAnnealingLR(alpha_optimizer, total_epoch)
weight_optimizer = optim.SGD(weight_list, lr=args.weight_lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
weight_scheduler = optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, total_epoch)


train_batch = len(train_loader)
test_batch = len(test_loader)
ShapeAdaptor.penalty = 1.0  # no penalty required in autotl mode
iteration = 0
avg_cost = np.zeros([total_epoch, 4], dtype=np.float32)
shape_list = []
sigmoid_alpha_list = []
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
        if i % args.step == 0:
            train_pred = model(train_data)
            train_loss = model_fit(train_pred, train_label)
            train_loss.backward()
            alpha_optimizer.step()

            weight_optimizer.zero_grad()
            alpha_optimizer.zero_grad()

        # update weight with updated alphas
        train_pred = model(train_data)
        train_loss = model_fit(train_pred, train_label)
        train_loss.backward()
        weight_optimizer.step()

        weight_optimizer.zero_grad()
        alpha_optimizer.zero_grad()

        # calculate training logging and accuracy
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
    alpha_scheduler.step()

    # compute memory and parameter usage
    input_data = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(input_data, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print('EPOCH: {:04d} ITER: {:04d} | TRAIN [LOSS|ACC.]: {:.4f} {:.4f} || TEST [LOSS|ACC.]: {:.4f} {:.4f} || MACs {} Params {}'
          .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2], avg_cost[index][3], flops, params))

    alphas = [0.5 + 0.5 * torch.sigmoid(i).squeeze().detach().cpu().numpy() for i in alpha_list]
    sigmoid_alpha_list.append(alphas)
    shape_list.append(model.shape_list)
    print('sigmoid(alpha) = {} | current shape = {}'.format(sigmoid_alpha_list[-1], shape_list[-1]))
    print('TOP: {}'.format(max(avg_cost[:, 3])))

end_time = time.time()
print('Total training takes {:.4f} seconds.'.format(end_time - start_time))

dict = {'shape-list': shape_list,
        'alpha': sigmoid_alpha_list,
        'loss': avg_cost,
        'macs': flops,
        'parms': params,
        'time': end_time - start_time}

np.save('logging/autotl_{}_{}.npy'
        .format(args.dataset, args.id), dict)
