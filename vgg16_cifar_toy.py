import torch
import torch.nn as nn
import numpy as np

from thop import profile
from thop import clever_format
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK'] ='True'
parser = argparse.ArgumentParser(description='VGG-16 on CIFAR-100: A toy example')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--mode', default='shape-adaptor', type=str, help='human, shape-adaptor')

args = parser.parse_args()


class VGG16(nn.Module):
    def __init__(self, mode='shape-adaptor'):
        super(VGG16, self).__init__()
        filter = [64, 128, 256, 512, 512]
        self.mode = mode
        self.shape_list = []

        # define VGG-16 block
        self.block1_1 = self.conv_layer(3, filter[0])
        self.block1_2 = self.conv_layer(filter[0], filter[0])

        self.block2_1 = self.conv_layer(filter[0], filter[1])
        self.block2_2 = self.conv_layer(filter[1], filter[1])

        self.block3_1 = self.conv_layer(filter[1], filter[2])
        self.block3_2 = self.conv_layer(filter[2], filter[2])
        self.block3_3 = self.conv_layer(filter[2], filter[2])

        self.block4_1 = self.conv_layer(filter[2], filter[3])
        self.block4_2 = self.conv_layer(filter[3], filter[3])
        self.block4_3 = self.conv_layer(filter[3], filter[3])

        self.block5_1 = self.conv_layer(filter[3], filter[4])
        self.block5_2 = self.conv_layer(filter[4], filter[4])
        self.block5_3 = self.conv_layer(filter[4], filter[4])

        # define classifier
        self.classifier = nn.Sequential(
            nn.Linear(filter[-1], 100),
        )

        self.criterion = nn.CrossEntropyLoss()

        self.pooling = nn.MaxPool2d(2, 2)

        # initialisations computed by; d_in = 32 and d_out = 8
        self.alpha = nn.Parameter(-0.346 * torch.ones(4, requires_grad=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def shape_adaptor(self, input, alpha):
        sigmoid_alpha = torch.sigmoid(alpha)
        s_alpha = 0.5 * sigmoid_alpha.item() + 0.5

        # use local-type shape adaptors
        input1_rs = F.interpolate(self.pooling(input), scale_factor=2 * s_alpha, mode='bilinear', align_corners=True)
        input2_rs = F.interpolate(input, size=input1_rs.shape[-2:], mode='bilinear', align_corners=True)

        return (1 - sigmoid_alpha) * input1_rs + sigmoid_alpha * input2_rs

    def conv_layer(self, in_channel, out_channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        return conv_block

    def forward(self, x):
        self.shape_list = []

        # VGG-16 Block 1
        self.shape_list.append(x.shape[-1])
        x = self.block1_1(x)

        self.shape_list.append(x.shape[-1])
        if self.mode == 'shape-adaptor':
            x = self.shape_adaptor(self.block1_2(x), self.alpha[0])
        else:
            x = self.pooling(self.block1_2(x))

        # VGG-16 Block 2
        self.shape_list.append(x.shape[-1])
        x = self.block2_1(x)

        self.shape_list.append(x.shape[-1])
        if self.mode == 'shape-adaptor':
            x = self.shape_adaptor(self.block2_2(x), self.alpha[1])
        else:
            x = self.pooling(self.block2_2(x))

        # VGG-16 Block 3
        self.shape_list.append(x.shape[-1])
        x = self.block3_1(x)

        self.shape_list.append(x.shape[-1])
        x = self.block3_2(x)

        if self.mode == 'shape-adaptor':
            x = self.shape_adaptor(self.block3_3(x), self.alpha[2])
        else:
            x = self.pooling(self.block3_3(x))

        # VGG-16 Block 4
        self.shape_list.append(x.shape[-1])
        x = self.block4_1(x)

        self.shape_list.append(x.shape[-1])
        x = self.block4_2(x)

        self.shape_list.append(x.shape[-1])
        if self.mode == 'shape-adaptor':
            x = self.shape_adaptor(self.block4_3(x), self.alpha[3])
        else:
            x = self.pooling(self.block4_3(x))

        # VGG-16 Block 5
        self.shape_list.append(x.shape[-1])
        x = self.block5_1(x)

        self.shape_list.append(x.shape[-1])
        x = self.block5_2(x)

        self.shape_list.append(x.shape[-1])
        x = self.block5_3(x)

        # task-prediction layer
        x = F.adaptive_avg_pool2d(x, 1)
        pred = self.classifier(x.view(x.size(0), -1))
        return pred


# define image transformation and dataset
batch_size = 128
trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
])
trans_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
])

train_set = dset.CIFAR100(root='dataset', train=True, transform=trans_train)
test_set  = dset.CIFAR100(root='dataset', train=False, transform=trans_test)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)


# define model
device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")
model = VGG16(args.mode).to(device)

# define individual parameter lists for network shape and network weight
alpha_list = []
weight_list = []
for _, (key, value) in enumerate(model.named_parameters()):
    if 'alpha' in key:
        alpha_list.append(value)
    else:
        weight_list.append(value)

total_epoch = 200
if args.mode == 'shape-adaptor':
    alpha_optimizer = optim.SGD(alpha_list, lr=0.1, momentum=0.9, nesterov=True)
    alpha_scheduler = optim.lr_scheduler.CosineAnnealingLR(alpha_optimizer, total_epoch)

weight_optimizer = optim.SGD(weight_list, lr=0.1, weight_decay=5e-4, momentum=0.9, nesterov=True)
weight_scheduler = optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, total_epoch)


train_batch = len(train_loader)
test_batch = len(test_loader)
iteration = 0
avg_cost = np.zeros([total_epoch, 4], dtype=np.float32)
shape_list = []
sigmoid_alpha_list = []
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
        if args.mode == 'shape-adaptor':
            train_pred = model(train_data)
            train_loss = model.criterion(train_pred, train_label)
            train_loss.backward()
            alpha_optimizer.step()

            weight_optimizer.zero_grad()
            alpha_optimizer.zero_grad()

        # update weight with updated alphas
        train_pred = model(train_data)
        train_loss = model.criterion(train_pred, train_label)
        train_loss.backward()
        weight_optimizer.step()

        weight_optimizer.zero_grad()
        if args.mode == 'shape-adaptor':
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

            test_pred = model(test_data)
            test_loss = model.criterion(test_pred, test_label)

            # compute test data accuracy
            test_predict_label1 = test_pred.data.max(1)[1]
            test_acc1 = test_predict_label1.eq(test_label).sum().item() / test_data.shape[0]

            cost[0] = torch.mean(test_loss).item()
            cost[1] = test_acc1
            avg_cost[index][2:] += cost / test_batch

    # scheduler update
    weight_scheduler.step()
    if args.mode == 'shape-adaptor':
        alpha_scheduler.step()

    # compute memory and parameter usage
    input_data = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(model, inputs=(input_data, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print('EPOCH: {:04d} ITER: {:04d} | TRAIN [LOSS|ACC.]: {:.4f} {:.4f} || TEST [LOSS|ACC.]: {:.4f} {:.4f} || MACs {} Params {}'
          .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2], avg_cost[index][3], flops, params))

    if args.mode == 'shape-adaptor':
        alphas = [0.5 + 0.5 * torch.sigmoid(i).squeeze().detach().cpu().numpy() for i in alpha_list]
        print('s(alpha) = {} | current shape = {}'.format(alphas, model.shape_list))
    else:
        print('human designed shape = {}'.format(model.shape_list))
    print('TOP: {}'.format(max(avg_cost[:, 3])))


