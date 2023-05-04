import torch.nn as nn
import torch
import torch.functional as F

class GooLenet(nn.Module):
    def __init__(self, num_classes = 1000, aux_logits = True, init_weights = False):
        super(GooLenet, self).__init__()
        # 要不要使用辅助分类器
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size = 7, stride = 2, padding = 3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode= True)

        self.conv2 = BasicConv2d(64, 64, kernel_size = 1)
        self.conv3 = BasicConv2d(64, 192, kernel_size = 3, padding = 1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode= True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode = True)

        self.inception4a = Inception()


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))
