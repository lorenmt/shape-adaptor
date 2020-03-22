import numpy as np
import scipy.io
from os import path
import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
# import tikzplotlib
import imageio


index = [4, 8, 17, 22]
color = [[0.89, 0, 0.263], [0.925, 0.451, 0], [0.396, 0.188, 0.596], [0.40, 0.643, 0.039]]
plt.figure(figsize=(5, 2))
for i in range(4):
    a = np.load('logging/aircraft_mobilenetv2_shape-adaptor_sanum6_ouputdim{}_0.npy'.format(index[i]), allow_pickle=True)
    b = np.load('logging/aircraft_mobilenetv2_shape-adaptor_sanum6_ouputdim{}_1.npy'.format(index[i]), allow_pickle=True)
    a1 = [k[5] for k in a.item()['alpha']]
    b1 = [k[5] for k in b.item()['alpha']]
    plt.plot((np.asarray(a1)+np.asarray(b1))/2, color=color[i], lw=5)
    plt.fill_between(np.arange(0, 200, 1), a1, b1, facecolor=(color[i]+[0.5]))
plt.xlim((0, 200))
plt.xticks([0, 50, 100, 150, 200])
plt.ylim((0.5, 1.0))
plt.yticks([0.5, 0.75, 1.0])
plt.xlabel('Epochs')
plt.ylabel('s(alpha)')
plt.show()
tikzplotlib.save("../paper/image/mobilenetv2_aircraft_5.tex")

#######################
a = np.load('logging/autonc_cifar100_vgg_shape-adaptor_sanum5_outputdim2_0.npy', allow_pickle=True)
shape_total = a.item()['shape-list']
aa = 0
figs = []
import matplotlib.font_manager as fm
prop = fm.FontProperties(fname='/Users/shikunliu/Library/Fonts/texgyrepagella-regular.otf')
for shape in shape_total:
    y = 0
    vgg_len = 13
    sa_num = len(shape) - 1
    shape_index = [int(i * vgg_len / sa_num) for i in range(sa_num)]
    # shape_index = [1,3,6,9]
    index = 0
    plt.text(0, 1.2, 'Epoch {}'.format(aa), horizontalalignment='center', fontproperties=prop, size=18)
    for i in range(vgg_len):
        if (i - 1) in shape_index:
            index += 1
        scale = shape[index] / 32
        # scale = 0.5 ** index
        x = 0 - (10 * scale) / 2
        rectangle = plt.Rectangle((x, y), 10 * scale, 0.8, color=(0.70, 0.71, 0.71))
        y -= 1
        plt.gca().add_patch(rectangle)
        plt.axis('scaled')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('../paper/gif/autotl_vgg_cifar100_{}.png'.format(aa), bbox_inches='tight', pad_inches=0)
    aa += 1
    plt.close()

images = []
for i in range(300):
    file_path = os.path.join('../paper/gif/autotl_vgg_cifar100_{}.png'.format(i))
    images.append(imageio.imread(file_path))
imageio.mimsave('autonc.gif', images, fps=10)


################################################
list = [4, 6, 8]
for k in range(3):
    a = np.load('logging/cifar100_vgg_shape-adaptor_sanum{}_outputdim8_0.npy'.format(list[k]), allow_pickle=True)
    # shape = a.item()['shape-list']
    shape = [32, 16, 9, 6, 5]
    y = 0
    layer_len = 13
    # layer_len = 19
    # layer_len = 17
    sa_num = len(shape) - 1
    shape_index = [int(i * (layer_len - 1) / sa_num) for i in range(sa_num)]
    # shape_index = [0, 2, 4, 7, 14]
    # shape_index = [0, 2, ]
    index = 0
    for i in range(layer_len):
        if (i - 1) in shape_index:
            index += 1
        scale = shape[index] / 32
        # scale = 0.5 ** index
        x = 0 - (10 * scale) / 2
        rectangle = plt.Rectangle((x, y), 10 * scale, 3, color=(0.70, 0.71, 0.71))
        y -= 4
        plt.gca().add_patch(rectangle)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('../paper/image/vgg_cifar100_random_worst.png'.format(list[k]), bbox_inches='tight', pad_inches=0)
    plt.close()



################################################
shape = [32, 16, 8, 8, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 1, 1]
shape = [32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 4, 4]
# shape = [32, 32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 4, 4]
y = 0
for i in range(len(shape)):
    scale = shape[i] / 32
    x = 0 - (10 * scale) / 2
    rectangle = plt.Rectangle((x, y), 10 * scale, 3, color=(0.70, 0.71, 0.71))
    y -= 4
    plt.gca().add_patch(rectangle)
plt.axis('off')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('../eccv/image/resnet_cifar100_human-imagenet.png', bbox_inches='tight', pad_inches=0)
plt.close()

#################################################
import csv
data = np.ones((200, 2))
k = 0
with open('result22.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=' ')
    for row in readCSV:
        data[k, 0] = float(row[0][4:])
        data[k, 1] = float(row[2][11:])
        k = k + 1

plt.scatter(data[0:, 1], data[0:, 0]*100, color='r', label='Randomly Sampled')
plt.scatter(5.21e9, 79.39, color='b', label='Shape Adaptors Designed')
# plt.scatter(5.21e9, 0.7862, color='b')
plt.scatter(3.14e8, 75.39, color='y', label='Human Designed')

plt.xlim(2.8e8, 1e10)
plt.xscale('log')
plt.xticks([3e8, 5e8, 1e9, 2e9, 5e9, 1e10], ['300M', '500M', '1G', '2G', '5G', '10G'])
plt.legend(loc=4)
tikzplotlib.save("../paper/image/random_search2.tex")

#######################
perturb = ['gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur', 'spatter', 'brightness', 'translate', 'rotate', 'tilt', 'scale']
human         = [0.4411, 0.2736, 0.1699, 0.01093, 0.03982, 0.01346, 0.0401, 0.0611, 0.01944, 0.068]
maxblurpool   = [0.3977, 0.2548, 0.1558, 0.01022, 0.04068, 0.01472, 0.0399, 0.0547, 0.01878, 0.0651]
shape_adaptor = [0.3658, 0.2565, 0.1509, 0.00643, 0.03475, 0.01371, 0.0245, 0.0415, 0.01268, 0.03899]


plt.bar(np.arange(10)+0.25, width=0.25, height=shape_adaptor)
plt.bar(np.arange(10), width=0.25, height=maxblurpool)
plt.bar(np.arange(10)-0.25, width=0.25, height=human)
plt.xticks(np.arange(10), perturb, rotation=30, ha='right')
tikzplotlib.save("../paper/image/perturbation2.tex")

#############################

#######################
a = np.load('logging_2/cifar100_resnet_shape-adaptor_0.npy', allow_pickle=True)
b = np.load('logging_2/cifar100_resnet_human-cifar_1.npy', allow_pickle=True)
c = np.load('logging_2/cifar100_resnet_human-imagenet_0.npy', allow_pickle=True)
shape_total = a.item()['shape-list']
aa = 0
figs = []
import matplotlib.font_manager as fm
# prop = fm.FontProperties(fname='/Users/shikunliu/Library/Fonts/MinionPro-regular.otf')
prop = fm.FontProperties(fname='/home/shikun/Downloads/MinionPro-Regular.otf')
prop_italic = fm.FontProperties(fname='/home/shikun/Downloads/MinionPro-It.otf')
for shape in shape_total:
    plt.figure(figsize=(7, 4.5))
    y = 0
    resnet_len = 18
    sa_num = len(shape) - 1
    shape_index = [int(i * (resnet_len-1) / sa_num) for i in range(sa_num)]
    shape_cifar_index = [4, 8, 14]
    shape_imagenet_index = [0, 1, 5, 9, 15]
    index = 0; cifar_index = 0; imagenet_index = 0
    plt.text(-32, -22, 'Epoch {}'.format(aa), horizontalalignment='center', fontproperties=prop, size=13)
    plt.text(-12, -22, 'All ResNet-50 networks have the same parameter space.', horizontalalignment='center', fontproperties=prop_italic, size=13)
    plt.text(-12, 6, 'ResNet-50 on CIFAR-100', horizontalalignment='center', fontproperties=prop_italic, size=13, style='italic')

    plt.text(0, 1.5, 'Shape Adaptor\n Designed', horizontalalignment='center', fontproperties=prop, size=13)
    plt.text(-12, 1.5, 'Human Designed B\n (for CIFAR-100)', horizontalalignment='center', fontproperties=prop, size=13)
    plt.text(-24, 1.5, 'Human Designed A\n (for ImageNet)', horizontalalignment='center', fontproperties=prop, size=13)

    plt.text(0, -19, 'Acc: {:.02f}'.format(a.item()['loss'][:aa+1,3].max()*100), horizontalalignment='center', fontproperties=prop, size=13)
    plt.text(-12, -19, 'Acc: {:.02f}'.format(b.item()['loss'][:aa+1,3].max()*100), horizontalalignment='center', fontproperties=prop, size=13)
    plt.text(-24, -19, 'Acc: {:.02f}'.format(c.item()['loss'][:aa+1,3].max()*100), horizontalalignment='center', fontproperties=prop, size=13)
    for i in range(resnet_len):
        ## shape adaptor shape
        if i < 17:
            if (i - 1) in shape_index:
                index += 1
            scale = shape[index] / 32
            # scale = 0.5 ** index
            x = 0 - (10 * scale) / 2
            rectangle = plt.Rectangle((x, y-i/16), 10 * scale, 0.8, color=(0.70, 0.71, 0.71))
            plt.gca().add_patch(rectangle)

            if (i - 1) in shape_cifar_index:
                cifar_index += 1
            scale = 0.5 ** cifar_index
            x = 0 - (10 * scale) / 2 - 12
            rectangle = plt.Rectangle((x, y-i/16), 10 * scale, 0.8, color=(0.70, 0.71, 0.71))
            plt.gca().add_patch(rectangle)

        if (i - 1) in shape_imagenet_index:
            imagenet_index += 1
        scale = 0.5 ** imagenet_index
        x = 0 - (10 * scale) / 2 - 24
        rectangle = plt.Rectangle((x, y), 10 * scale, 0.8, color=(0.70, 0.71, 0.71))
        y -= 1
        plt.gca().add_patch(rectangle)
        plt.axis('scaled')
    plt.axis('off')

    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('../paper/gif_2/resnet50_{}.png'.format(aa))
    aa += 1
    plt.close()

###########################
#######################
a = np.load('logging_2/cifar100_resnet_shape-adaptor_0.npy', allow_pickle=True)
b = np.load('logging_2/cifar100_resnet_human-cifar_1.npy', allow_pickle=True)
c = np.load('logging_2/cifar100_resnet_human-imagenet_0.npy', allow_pickle=True)
figs = []
import matplotlib.font_manager as fm
# prop = fm.FontProperties(fname='/Users/shikunliu/Library/Fonts/MinionPro-regular.otf')
prop = fm.FontProperties(fname='/home/shikun/Downloads/MinionPro-Regular.otf')
prop_italic = fm.FontProperties(fname='/home/shikun/Downloads/MinionPro-It.otf')
for i in range(200):
    plt.figure(figsize=(5, 4.5))
    plt.plot(c.item()['loss'][:i+1,3]*100, label='Human Designed A', color='blue')
    plt.plot(b.item()['loss'][:i+1,3]*100, label='Human Designed B', color='red')
    plt.plot(a.item()['loss'][:i+1,3]*100, label='Shape Adaptor Designed', color='orange')

    plt.scatter(i, a.item()['loss'][i,3]*100, s=18, color='orange')
    plt.scatter(i, b.item()['loss'][i,3]*100, s=18, color='red')
    plt.scatter(i, c.item()['loss'][i,3]*100, s=18, color='blue')
    plt.xlim(0, 200)
    plt.ylim(5, 85)
    plt.ylabel('Accuracy', fontproperties=prop, size=13)
    plt.xlabel('Epochs', fontproperties=prop, size=13)
    plt.xticks(fontproperties=prop, size=13)
    plt.yticks(fontproperties=prop, size=13)
    plt.legend(loc='lower right', prop=prop,frameon=False)

    plt.savefig('../paper/gif_3/resnet50_{}.png'.format(i))
    plt.close()

images = []
for i in range(200):
    file_path = os.path.join('../paper/gif_2/resnet50_{}.png'.format(i))
    images.append(imageio.imread(file_path))
imageio.mimsave('../paper/resnet50_1.gif', images, fps=20)