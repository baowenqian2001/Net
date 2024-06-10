from .VGG import vgg11, vgg13, vgg16, vgg19 
from .ZFNet import zfnet

cfgs = {
    # 'alexnet': alexnet,
    'zfnet': zfnet,
    # 'vgg': vgg16,
    # 'vgg_tiny': vgg11,
    # 'vgg_small': vgg13,
    # 'vgg_big': vgg19,
}

def find_model_using_name(model_name, num_classes):
    return cfgs[model_name](num_classes)
