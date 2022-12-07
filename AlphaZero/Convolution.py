import torch.nn as nn


def build_convolution(in_channels, filter_size, kernel, momentum=.9, stride=(0, 0), padding=0):
    """
    Build a convolutional layer with the given number of channels and depth.
    :param padding:
    :param momentum:
    :param stride:
    :param in_channels: The number of input channels in the convolutional layer.
    :param filter_size: The number of filters applied to the input channels (output channels) in the convolutional layer.
    :param kernel: The kernel size of the convolutional layer.
    :return: A convolutional layer with the given number of channels.
    """
    conv = nn.Sequential()
    conv.add_module('conv', nn.Conv2d(in_channels, filter_size, kernel, stride=stride, padding=padding))
    conv.add_module('batch_norm', nn.BatchNorm2d(filter_size, momentum=momentum))
    conv.add_module('relu', nn.ReLU())

    return conv
