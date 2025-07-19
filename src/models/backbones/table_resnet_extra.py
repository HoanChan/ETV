import torch.nn as nn
from mmocr.registry import MODELS
from .resnet_extra import ResNetExtra

@MODELS.register_module()
class TableResNetExtra(ResNetExtra):
    def __init__(self, layers, input_dim=3, gcb_config=None, init_cfg=None):
        super().__init__(layers, input_dim, gcb_config, init_cfg)
        # Only override maxpool3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)