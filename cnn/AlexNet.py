import torch 
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classer=1000):
        super().__init__()