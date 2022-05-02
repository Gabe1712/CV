from os import remove
from ssd.modeling import AnchorBoxes
from ssd.modeling.backbones.basic import BasicModel
from tops.config import LazyCall as L


##
from ssd.data.transforms import (
    ToTensor, RandomHorizontalFlip, RandomSampleCrop, Normalize, Resize,
    GroundTruthBoxesToAnchors)
import torch
import torchvision
import getpass
import pathlib
from configs.utils import get_dataset_dir, get_output_dir
from ssd.data import MNISTDetectionDataset
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from ssd.modeling import SSD300, SSDMultiboxLoss, backbones
##


# The line belows inherits the configuration set for the tdt4265 dataset
from ..tdt4265 import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)

# The config below is copied from the ssd300.py model trained on images of size 300*300.
# The images in the tdt4265 dataset are of size 128 * 1024, so resizing to 300*300 is probably a bad idea
# Change the imshape to (128, 1024) and experiment with better prior boxes
train.imshape = (128, 1024)
train.epochs = 40
train.image_channels = 3
anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

##ADD FROM NOW ONE
backbone = L(backbones.BasicModel)(
    output_channels=[128, 256, 128, 128, 64, 64],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

optimizer.lr= 5e-3


train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(RandomHorizontalFlip)(),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])
val_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
])
gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(Normalize)(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
data_train.dataset = L(MNISTDetectionDataset)(
    data_dir=get_dataset_dir("data/mnist_object_detection/train"),  transform=train_cpu_transform, is_train=True,)

data_val.dataset = L(MNISTDetectionDataset)(
    data_dir=get_dataset_dir("data/mnist_object_detection/val"), transform=val_cpu_transform, is_train=False)
data_val.gpu_transform = gpu_transform
data_train.gpu_transform = gpu_transform

label_map =  {idx: cls_name for idx, cls_name in enumerate(MNISTDetectionDataset.class_names)}