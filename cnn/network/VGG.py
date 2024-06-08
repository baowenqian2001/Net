import torch 
import torch.nn as nn

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

# vgg_tiny(VGG11), vgg_small(VGG13), vgg(VGG16), vgg_big(VGG19)
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],   
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), # 第2线性层
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

        def forward(self, x):
            x = self.features(x)    # N x 3 x 224 x 224
            x = torch.flatten(x, start_dim=1)   # N x 512 x 7 x 7
            x = self.classifier(x)  # N x 512 x 7 x 7
            
            return x

def make_features(cfg):
    layers = [] 
    in_channels = 3 
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, Kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


# 全局算法
def vgg(model_name='vgg16',**kwargs):
    assert model_name in cfgs, "Warning: model unmber {} not in cfgs dist!".format(model_name)
    cfg = cfgs[model_name]
    model = VGG(make_features(cfg), **kwargs)

    return model
def vgg11(num_classes): 
    cfg = cfgs["vgg11"]
    model = VGG(make_features(cfg), num_classes=num_classes)
    return model

def vgg13(num_classes):  
    cfg = cfgs["vgg13"]
    model = VGG(make_features(cfg), num_classes=num_classes)
    return model

def vgg16(num_classes):  
    cfg = cfgs["vgg16"]
    model = VGG(make_features(cfg), num_classes=num_classes)
    return model

def vgg19(num_classes):  
    cfg = cfgs['vgg19']
    model = VGG(make_features(cfg), num_classes=num_classes)
    return model