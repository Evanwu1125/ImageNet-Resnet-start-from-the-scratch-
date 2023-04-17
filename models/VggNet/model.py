import torch.nn as nn
import torch

class Vgg(nn.Module):
    def __init__(self, features, num_classes = 1000, init_weights = False):
        super(Vgg, self).__init__()
        #vgg中的backbone网络根据选择采用不同的深度
        self.backbone = features
        #这里的分类器是最后三个全连接层组成的，其中最后一层全连接层的层数为图片的类别数
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        #x的形状应该是 N * 3 * 224 * 224
        #N代表批量大小，3代表通道数，224 * 224 代表图片的长和宽
        x = self.backbone(x)
        #backbone网络输出的x的形状为N * 512 * 7 * 7
        x = torch.flatten(x, start_dim = 1)
        #经过上面这个函数之后把图像拉成一个长的向量，方便输入到全连接层中
        x = self.classifier(x)
        return x

    #构造一个关于卷积层和线性层的初始化函数，不需要返回任何参数
    def _initalize_weights(self):
        for m in self.modules():
            #这里分别对卷积层和线性层进行初始化
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg):
    layers = []
    in_channel = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channel, v, kernel_size=3, padding=1), nn.ReLU()]
            #这里把当前卷积的通道数设置成下一个卷积核的输入通道
            in_channel = v
    #这里layers之前之所以要加*，是因为nn.Sequential不接受整个列表的输入，而是逐个的输入，
    #所以我们使用*号把参数变成非关键字传输，这样就满足了nn.Sequential的输入要求
    return nn.Sequential(*layers)

cfgs = {
    'vgg11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(model_name = 'vgg16', **kwargs):
    assert model_name in cfgs, "Warning: model does not exist"
    cfg = cfgs[model_name]

    model = vgg(make_layers(cfg), **kwargs)
    return model