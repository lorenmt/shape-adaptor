import torch
import torch.nn as nn
import numpy as np

from utils import *
import torch.nn.functional as F


def ShapeAdaptor(input1, input2, alpha, residual=False, r1=0.5, r2=1.0):
    # sigmoid_alpha = sigmoid(alpha) if having penalty, i.e. penalty = 1;
    # the penalty value will be defined/computed in the model_training file
    sigmoid_alpha = torch.sigmoid(alpha) * ShapeAdaptor.penalty + r1 / (r2 - r1) * (ShapeAdaptor.penalty - 1)
    s_alpha = (r2 - r1) * sigmoid_alpha.item() + r1

    # total no. of shape adaptors
    ShapeAdaptor.counter += 1

    # the true current dim without any penalty (will be used for computing the correct penalty value)
    ShapeAdaptor.current_dim_true *= ((r2 - r1) * torch.sigmoid(alpha).item() + r1)

    if ShapeAdaptor.type == 'local':
        # a shape adaptor will drop at least 1 dimension (local structure), used in standard or AutoTL mode
        ShapeAdaptor.current_dim = int(ShapeAdaptor.current_dim * s_alpha)
        dim = 1 if ShapeAdaptor.current_dim < 1 else ShapeAdaptor.current_dim   # output dim should be at least 1

    elif ShapeAdaptor.type == 'global':
        # a shape adaptor could maintain the same dimension (global structure), used in AutoSC mode
        ShapeAdaptor.current_dim = ShapeAdaptor.current_dim * s_alpha
        dim = 1 if ShapeAdaptor.current_dim < 1 else round(ShapeAdaptor.current_dim)  # output dim should be at least 1

    '''
    input1 = resizing(x, scale=r1);  input2 = resizing(x, scale=r2)
    It's important to debug/confirm your model design using these two different implementations.
    Implementation A:
    input2_rs = F.interpolate(input2, scale_factor=(1/r2)*s_alpha, mode='bilinear', align_corners=True)
    input1_rs = F.interpolate(input1, size=input2_rs.shape[-2:], mode='bilinear', align_corners=True)

    Implementation B:
    input1_rs = F.interpolate(input1, scale_factor=(1/r1)*s_alpha, mode='bilinear', align_corners=True)
    input2_rs = F.interpolate(input2, size=input1_rs.shape[-2:], mode='bilinear', align_corners=True)

    Those two implementations (along with an additional version below) should produce the same shape.
    Note: +- 1 dim change in intermediate layers is expected due to different rounding methods.
    '''

    input1_rs = F.interpolate(input1, size=dim, mode='bilinear', align_corners=True)
    input2_rs = F.interpolate(input2, size=dim, mode='bilinear', align_corners=True)
    if residual:  # to keep gradient magnitude consistent with standard residuals: f(x) + x
        return 2 * (1 - sigmoid_alpha) * input1_rs + 2 * sigmoid_alpha * input2_rs
    else:
        return (1 - sigmoid_alpha) * input1_rs + sigmoid_alpha * input2_rs


def SA_init(input_dim, output_dim, sa_num, r1=0.5, r2=1.0):
    # input_dim: input data dimension;  output_dim: output / last layer feature dimension
    # input_dim * s(sigmoid(alpha)) ^ sa_num = output_dim, find alpha
    # s(sigmoid(alpha)) = r1 + (r2 - r1) * sigmoid(alpha)
    eps = 1e-4  # avoid inf
    if input_dim * r1 ** sa_num > output_dim:
        return np.log(eps)
    else:
        return np.log(-(np.power(output_dim / input_dim, 1.0/sa_num) - r1) / (np.power(output_dim / input_dim, 1.0/sa_num) - r2) + eps)


"""
VGG Network
"""


class VGG(nn.Module):
    def __init__(self, input_shape=32, output_shape=8, dataset=None, mode=None, sa_num=None, type='D'):
        super(VGG, self).__init__()
        self.dataset = dataset
        self.mode = mode
        self.sa_num = sa_num
        self.input_shape = input_shape
        self.shape_list = []
        self.filter = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
        }

        # define VGG feature extractor layers
        if self.input_shape > 64:
            # human-designed network will attach another max-pool layer before classifier (in large-resolution datasets)
            self.filter[type].append('M')

        layers = []
        channel_in = 3  # input RGB images
        self.features = nn.ModuleList()
        for ch in self.filter[type]:
            if ch == 'M':
                # AutoSC mode is built on the human-designed network *along with* the original resizing layers
                if 'human' in self.mode:
                    layers += [nn.MaxPool2d(2, 2)]
                elif self.mode == 'autosc':
                    layers += [nn.MaxPool2d(2, 2, ceil_mode=True)]  # use ceil mode to avoid 0 dimension feature layer
            else:
                # Standard mode is built on the human-designed network *without* the original resizing layers
                layers += [nn.Conv2d(channel_in, ch, kernel_size=3, padding=1),
                           nn.BatchNorm2d(ch),
                           nn.ReLU(inplace=True)]
                channel_in = ch
        self.features = nn.Sequential(*layers)

        # define two types of shape adaptor modes:
        if self.mode == 'shape-adaptor':
            ShapeAdaptor.type = 'local'
            self.max_pool = nn.MaxPool2d(2, 2)  # max-pool is considered as the down-sample branch.
            # we don't apply shape adaptor at the last layer, thus "-3": -1 * 3 operations in each conv layer.
            self.sampling_index_full = [i for i in range(len(self.features) - 3) if isinstance(self.features[i], nn.ReLU)]

            if self.sa_num is None:
                # automatically define the optimal number of shape adaptors based on a heuristic.
                self.sa_num = int(np.log2(self.input_shape / 2))

            # insert shape adaptors uniformly
            index_gap = len(self.sampling_index_full) / self.sa_num
            self.sampling_index = [self.sampling_index_full[int(i * index_gap)] for i in range(self.sa_num)]

        elif self.mode == 'autosc':
            ShapeAdaptor.type = 'global'
            self.max_pool = nn.MaxPool2d(2, 2, ceil_mode=True)  # use ceil mode to avoid 0 dimension feature layer
            # we don't insert shape adaptors on top of max-pooling layer. (excessive reshaping at the same position)
            self.sampling_index_full = [i for i in range(len(self.features)-3) if isinstance(self.features[i], nn.ReLU)
                                        and not isinstance(self.features[i+1], nn.MaxPool2d)]

            if self.sa_num is None:
                # number of shape adaptors found by a grid search, this number is possibly not optimal
                self.sa_num = 2 if self.input_shape < 64 else 4

            # insert shape adaptors uniformly
            index_gap = len(self.sampling_index_full) / self.sa_num
            self.sampling_index = [self.sampling_index_full[int(i * index_gap)] for i in range(self.sa_num)]

        # define fully-connected prediction layers; we use one fc-layer across all networks for consistency
        self.classifier = nn.Sequential(
            nn.Linear(512, CLASS_NB[dataset]),
        )

        if 'human' not in self.mode:
            if self.mode == 'shape-adaptor':
                # compute shape adaptor initialisation by a heuristic
                self.alpha = nn.Parameter(torch.tensor([SA_init(input_shape, output_shape, self.sa_num)] * self.sa_num, requires_grad=True))
            elif self.mode == 'autosc':
                # initialise shape adaptors to be consistent with the original human-designed network shape:
                # s(alpha) = 0.95, alpha = 2.19
                self.alpha = nn.Parameter(torch.tensor([2.19] * self.sa_num, requires_grad=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # randomly initialise network weights
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

    def forward(self, x):
        self.shape_list = []  # record spatial dimension in each layer for shape visualisation
        if 'human' in self.mode:
            for i in range(len(self.features)):
                if isinstance(self.features[i], nn.Conv2d):
                    self.shape_list.append(x.shape[-1])
                x = self.features[i](x)
        else:
            ShapeAdaptor.counter = 0
            ShapeAdaptor.current_dim = self.input_shape
            ShapeAdaptor.current_dim_true = self.input_shape
            for i in range(len(self.features)):
                if isinstance(self.features[i], nn.Conv2d):
                    self.shape_list.append(x.shape[-1])
                if isinstance(self.features[i], nn.MaxPool2d):
                    # in AutoSC mode, we need to include human-defined resizing layers to
                    # re-compute the correct current dimension for shape adaptors
                    ShapeAdaptor.current_dim = ShapeAdaptor.current_dim * 0.5
                    ShapeAdaptor.current_dim_true = ShapeAdaptor.current_dim * 0.5
                x = self.features[i](x)
                if i in self.sampling_index:
                    x = ShapeAdaptor(self.max_pool(x), x, self.alpha[ShapeAdaptor.counter])

        if self.input_shape > 64:
            # include the last max-pooling layer
            self.shape_list.append(x.shape[-1])

        output = self.avg_pool(x)
        pred = self.classifier(output.view(output.size(0), -1))
        return pred


"""
ResNet Network
"""


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64, apply_sa=False, input_shape=32, output_shape=8, sa_num=None):
        super(BasicBlock, self).__init__()

        # self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.apply_sa = apply_sa
        BasicBlock.counter += 1

        if self.apply_sa:
            self.alpha = nn.Parameter(torch.tensor(SA_init(input_shape, output_shape, sa_num), requires_grad=True))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.apply_sa:
            out = ShapeAdaptor(self.downsample(x), out, self.alpha, residual=True)
        else:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64, apply_sa=False, input_shape=None, output_shape=8, sa_num=None, alpha=None):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        # self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.apply_sa = apply_sa

        Bottleneck.counter += 1

        if self.apply_sa:
            if alpha is None:
                self.alpha = nn.Parameter(torch.tensor(SA_init(input_shape, output_shape, sa_num), requires_grad=True))
            else:
                self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.apply_sa:
            out = ShapeAdaptor(self.downsample(x), out, self.alpha, residual=True)
        else:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, dataset, mode, input_shape=32, output_shape=8, sa_num=None):
        super(ResNet, self).__init__()
        self.dataset = dataset
        self.mode = mode
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.sa_num = sa_num
        self.shape_list = []
        self.inplanes = 64
        self.base_width = 64
        self.layers = layers

        if self.mode == 'human-imagenet' or (self.mode == 'autosc' and self.input_shape >= 64):
            # large dataset using a 7 x 7 conv layer (the original ResNet for ImageNet)
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.mode == 'human-cifar' or (self.mode == 'autosc' and self.input_shape < 64):
            # small dataset using a 3 x 3 conv layer (typically used in CIFAR-like small-resolution datasets)
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1   = nn.BatchNorm2d(self.inplanes)
        self.relu  = nn.ReLU(inplace=True)
        self.block = block
        self.block.counter = 1 if self.mode != 'autosc' else 0

        if self.mode in ['shape-adaptor', 'autotl']:
            ShapeAdaptor.type = 'local'
            if self.sa_num is None:
                # optimal number of shape adaptors is computed by a heuristic
                self.sa_num = int(np.log2(self.input_shape / 2))
            # include the first conv layer, and not the last layer
            self.index_gap = (sum(layers) + 1 - 1) / self.sa_num

            if self.input_shape > 64:
                # choose kernel size for the first conv layer as consistent from human-designed networks
                self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
            else:
                self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.MaxPool2d(2, 2)

            if self.mode == 'shape-adaptor':
                self.alpha_init = nn.Parameter(torch.tensor(SA_init(input_shape, output_shape, self.sa_num), requires_grad=True))

            elif self.mode == 'autotl':
                # in AutoTL mode, we replace human-defined resizing layers by shape adaptors
                # the original ResNet has two initial down-sampling layers before residual blocks
                # s(alpha) = 0.55 -> alpha = -2.19,
                # initialise shape adaptors to make them consistent with the original human-designed network shape
                self.alpha_init1 = nn.Parameter(torch.tensor(-2.19, requires_grad=True))
                self.alpha_init2 = nn.Parameter(torch.tensor(-2.19, requires_grad=True))

        elif self.mode == 'autosc':
            ShapeAdaptor.type = 'global'
            if self.sa_num is None:
                # number of shape adaptors found by a grid search, this number is possibly not optimal
                self.sa_num = 2 if self.input_shape <= 64 else 4
            # not include human-defined resizing layers (3) and the last layer (1)
            self.index_gap = (sum(layers) - 3 - 1) / self.sa_num

        # official ResNet structure design
        self.layer1 = self._make_layer(block, 64,  self.layers[0])
        self.layer2 = self._make_layer(block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, self.layers[3], stride=2, endlayer=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, CLASS_NB[dataset])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, endlayer=False):
        if endlayer:
            blocks = blocks - 1
        layers = []

        if 'human' in self.mode:
            # original human-designed ResNets
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers.append(block(self.inplanes, planes, stride, downsample, self.base_width))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, base_width=self.base_width))

        elif self.mode == 'autosc':
            # in AutoSC mode, we include the original down-sampling layers in the original ResNet design
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers.append(block(self.inplanes, planes, stride, downsample, self.base_width))

            # down-sampling layer does not include in layer counter
            # (not to be stacked with shape adaptors for excessive resizing)
            self.block.counter -= 1
            self.inplanes = planes * block.expansion

            for _ in range(1, blocks):
                if self.block.counter in [int(i * self.index_gap) for i in range(self.sa_num)]:
                    downsample = nn.Sequential(
                        conv1x1(self.inplanes, planes * block.expansion, stride=2),
                        nn.BatchNorm2d(planes * block.expansion),
                    )
                    # residual shape adaptors: weight layer as identity branch, 1 x 1 conv as down-sampling branch
                    layers.append(block(self.inplanes, planes, 1, downsample, self.base_width, True, self.input_shape,
                                        self.output_shape, self.sa_num, alpha=2.19))
                else:
                    # if not resizing, just add standard residual connections
                    layers.append(block(self.inplanes, planes, base_width=self.base_width))

        elif self.mode == 'autotl':
            # in AutoTL mode, we replace all human-defined resizing layers into shape adaptors
            downsample = None
            if stride == 1 and self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
                layers.append(block(self.inplanes, planes, stride, downsample, self.base_width))

            elif stride > 1:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=2),
                    nn.BatchNorm2d(planes * block.expansion),
                )
                # s(alpha) = 0.55 -> alpha = -2.19, for near 0.5 resizing
                layers.append(block(self.inplanes, planes, 1, downsample, self.base_width, True, self.input_shape,
                                    self.output_shape, self.sa_num, alpha=-2.19))
            else:
                # if not resizing, just add standard residual connections
                layers.append(block(self.inplanes, planes, stride, downsample, self.base_width))

            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, base_width=self.base_width))

        elif self.mode == 'shape-adaptor':
            # standard shape adaptor networks: only shape adaptors contribute to the resizing in the entire network
            for layer_index in range(blocks):
                if layer_index == 0:
                    # if in shape adaptor index, insert a shape adaptor here:
                    # identity block: weight layer; down-sampling block: 1 x 1 conv layer
                    if self.block.counter in [int(i * self.index_gap) for i in range(1, self.sa_num)]:
                        downsample = nn.Sequential(
                            conv1x1(self.inplanes, planes * block.expansion, stride=2),
                            nn.BatchNorm2d(planes * block.expansion),
                        )
                        layers.append(block(self.inplanes, planes, 1, downsample, self.base_width, True, self.input_shape, self.output_shape, self.sa_num))
                    else:
                        # the first layer in any block always uses an 1 x 1 down-sampling (expand feature dimension)
                        # to be consistent with human-designed network
                        downsample = nn.Sequential(
                            conv1x1(self.inplanes, planes * block.expansion, stride=1),
                            nn.BatchNorm2d(planes * block.expansion),
                        )
                        layers.append(block(self.inplanes, planes, 1, downsample, self.base_width))
                    self.inplanes = planes * block.expansion

                elif self.block.counter in [int(i * self.index_gap) for i in range(1, self.sa_num)]:
                    # similar thing applied to all other layers in a residual block
                    downsample = nn.Sequential(
                        conv1x1(self.inplanes, planes * block.expansion, stride=2),
                        nn.BatchNorm2d(planes * block.expansion),
                    )
                    layers.append(block(self.inplanes, planes, 1, downsample, self.base_width, True, self.input_shape, self.output_shape, self.sa_num))
                else:
                    # if not resizing, just add standard residual connections
                    layers.append(block(self.inplanes, planes, base_width=self.base_width))

        if endlayer:
            # final layer will not be inserted shape adaptors and would not be applied any width multiplier
            layers.append(block(self.inplanes, planes, base_width=self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        ShapeAdaptor.counter = 0
        ShapeAdaptor.current_dim = self.input_shape
        ShapeAdaptor.current_dim_true = self.input_shape

        # first two conv layers before residual blocks
        if self.mode == 'shape-adaptor':
            # initialise a list for spatial dimension for shape visualisation
            self.shape_list = [x.shape[-1]]
            x = self.relu(self.bn1(self.conv1(x)))
            x = ShapeAdaptor(self.maxpool(x), x, self.alpha_init)

        elif self.mode == 'autotl':
            # initialise a list for spatial dimension for shape visualisation
            self.shape_list = [x.shape[-1]]
            x = self.relu(self.bn1(self.conv1(x)))
            x = ShapeAdaptor(self.maxpool(x), x, self.alpha_init1)
            self.shape_list.append(x.shape[-1])
            x = ShapeAdaptor(self.maxpool(x), x, self.alpha_init2)

        elif self.mode == 'human-imagenet' or (self.mode == 'autosc' and self.input_shape > 64):
            # initialise a list for spatial dimension for shape visualisation
            self.shape_list = [x.shape[-1]]
            x = self.relu(self.bn1(self.conv1(x)))
            self.shape_list.append(x.shape[-1])
            x = self.maxpool(x)
            if self.mode == 'autosc':
                ShapeAdaptor.current_dim = self.input_shape * 0.25
                ShapeAdaptor.current_dim_true = self.input_shape * 0.25

        elif self.mode == 'human-cifar' or (self.mode == 'autosc' and self.input_shape <= 64):
            # initialise a list for spatial dimension for shape visualisation
            self.shape_list = [x.shape[-1]]
            x = self.relu(self.bn1(self.conv1(x)))

        # residual layers
        # in AutoSC mode, we need to recompute current dimension by including human-defined resizing layers
        for i in range(len(self.layer1)):
            self.shape_list.append(x.shape[-1])
            x = self.layer1[i](x)

        for i in range(len(self.layer2)):
            self.shape_list.append(x.shape[-1])
            if i == 0 and self.mode == 'autosc':
                ShapeAdaptor.current_dim *= 0.5
                ShapeAdaptor.current_dim_true *= 0.5
            x = self.layer2[i](x)

        for i in range(len(self.layer3)):
            if i == 0 and self.mode == 'autosc':
                ShapeAdaptor.current_dim *= 0.5
                ShapeAdaptor.current_dim_true *= 0.5
            self.shape_list.append(x.shape[-1])
            x = self.layer3[i](x)

        for i in range(len(self.layer4)):
            if i == 0 and self.mode == 'autosc':
                ShapeAdaptor.current_dim *= 0.5
                ShapeAdaptor.current_dim_true *= 0.5
            self.shape_list.append(x.shape[-1])
            x = self.layer4[i](x)

        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        return x


"""
MobileNetv2 Network
"""


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, apply_sa=False, input_shape=None, output_shape=16, sa_num=None, mode=None, alpha=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.apply_sa = apply_sa
        self.inp = inp
        self.oup = oup
        self.mode = mode

        if self.apply_sa:
            if alpha is None:
                self.alpha = nn.Parameter(torch.tensor(SA_init(input_shape, output_shape, sa_num), requires_grad=True))
            else:
                self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True))

        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.max_pool = nn.MaxPool2d(2, 2) if self.mode != 'autosc' else nn.MaxPool2d(2, 2, ceil_mode=True)

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.apply_sa:
            x = self.conv(x)
            return ShapeAdaptor(self.max_pool(x), x, self.alpha)
        else:
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, dataset=None, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, mode=None,
                 input_shape=32, output_shape=16, sa_num=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        self.mode = mode
        self.dataset = dataset
        self.input_shape = input_shape
        self.sa_num = sa_num
        self.shape_list = []

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2] if self.input_shape > 64 else [6, 24, 2, 1],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.down_sampling_index = []

        # insert shape adaptors:
        # include the first but not the last conv layers (defined outside the residual setting)
        self.layers_num = sum(i[2] for i in inverted_residual_setting) + 2 - 1
        if self.mode == 'shape-adaptor':
            ShapeAdaptor.type = 'local'
            if self.sa_num is None:
                self.sa_num = int(np.log2(self.input_shape / 2))
            self.index_gap = self.layers_num / self.sa_num

        elif self.mode == 'autosc':
            ShapeAdaptor.type = 'global'
            if self.sa_num is None:
                self.sa_num = 3 if self.input_shape < 64 else 4
            down_sampling_num = 3 if self.input_shape < 64 else 4
            self.index_gap = (self.layers_num - down_sampling_num - 2) / self.sa_num  # not include down-sampling layers

        if 'human' in self.mode:
            # changes for CIFAR-like small datasets based on the implementation here:
            # https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py
            if 'cifar' in self.mode:
                features = [ConvBNReLU(3, input_channel, stride=1)]
            else:
                features = [ConvBNReLU(3, input_channel, stride=2)]

            # building inverted residual blocks
            for t, c, n, s in inverted_residual_setting:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    input_channel = output_channel

        if self.mode == 'shape-adaptor':
            # the first conv layer before residuals
            self.alpha_init = nn.Parameter(torch.tensor(SA_init(input_shape, output_shape, self.sa_num), requires_grad=True))
            features = [ConvBNReLU(3, input_channel, stride=1)]
            self.max_pool = nn.MaxPool2d(2, 2)

            # standard mode for uniformly inserting shape adaptors
            count_layer = 1  # include the first layer
            for t, c, n, s in inverted_residual_setting:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    if count_layer in [int(i * self.index_gap) for i in range(1, self.sa_num)]:
                        features.append(block(input_channel, output_channel, 1, t, True, input_shape, output_shape,
                                              self.sa_num))
                    else:
                        features.append(block(input_channel, output_channel, 1, t))
                    count_layer += 1
                    input_channel = output_channel

        if self.mode == 'autosc':
            if self.input_shape <= 64:
                features = [ConvBNReLU(3, input_channel, stride=1)]
            else:
                features = [ConvBNReLU(3, input_channel, stride=2)]
            count_layer = -1  # does not include the early conv layer, we insert shape adaptors in the residual blocks

            # include resizing in the first conv layer; will be used to re-compute resizing for shape adaptors
            self.down_sampling_index = [0] if self.input_shape > 64 else []

            for t, c, n, s in inverted_residual_setting:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    if stride == 2:
                        features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                        input_channel = output_channel
                        self.down_sampling_index.append(len(features) - 1)
                        continue  # avoid layer counting

                    elif count_layer in [int(i * self.index_gap) for i in range(self.sa_num)]:
                        features.append(block(input_channel, output_channel, 1, t, True, input_shape, output_shape,
                                              self.sa_num, mode=self.mode, alpha=2.19))
                    else:
                        features.append(block(input_channel, output_channel, 1, t))
                    count_layer += 1
                    input_channel = output_channel

        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.last_channel, CLASS_NB[dataset]),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        ShapeAdaptor.counter = 0
        ShapeAdaptor.current_dim = self.input_shape
        ShapeAdaptor.current_dim_true = self.input_shape
        self.shape_list = [x.shape[-1]]  # for shape visualisation

        if 'human' in self.mode or self.mode == 'autosc':
            x = self.features[0](x)
            # this means you are applying AutoSC on large dataset;
            # thus to include the resizing in the first conv layer for computing the current dimension
            if 0 in self.down_sampling_index:
                ShapeAdaptor.current_dim = ShapeAdaptor.current_dim * 0.5
                ShapeAdaptor.current_dim_true = ShapeAdaptor.current_dim_true * 0.5
        else:
            x = self.features[0](x)
            x = ShapeAdaptor(self.max_pool(x), x, self.alpha_init)

        for i in range(1, len(self.features)):
            self.shape_list.append(x.shape[-1])
            if i in self.down_sampling_index:
                ShapeAdaptor.current_dim = ShapeAdaptor.current_dim * 0.5
                ShapeAdaptor.current_dim_true = ShapeAdaptor.current_dim_true * 0.5
            x = self.features[i](x)

        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


