# Shape Adaptor
This repository contains the source code to support the paper: [Shape Adaptor: A Learnable Resizing Module](https://arxiv.org). 
![alt text](resnet50.gif "Logo Title Text 1")


## Usage
`model_list.py` contains original pytorch implementation on VGG and ResNet with the option using human designed pooling layers and our shape adaptors.
`training_joint.py` joint training method by update alpha and network weights together in the same step.
`training_alternate.py` joint training method by update alpha and network weights separately in the different step.

The same training schedule applied in `imagenet` training files.

## Flags


