import torch.nn as nn

from mmocr.registry import MODELS
from mmcv.cnn.bricks import ContextBlock
from mmdet.models.backbones.resnet import BasicBlock

def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """ 1x1 convolution """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def get_gcb_config(gcb_config, layer):
    if gcb_config is None or not gcb_config['layers'][layer]:
        return None
    else:
        return gcb_config

@MODELS.register_module()
class ResNetExtra(nn.Module):

    def __init__(self, 
                 layers, 
                 input_dim=3, 
                 gcb_config=None,
                 init_cfg=None):
        assert len(layers) >= 4

        super(ResNetExtra, self).__init__()
        self.init_cfg = init_cfg
        self.inplanes = 128
        
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(BasicBlock, 256, layers[0], stride=1, gcb_config=get_gcb_config(gcb_config, 0))

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = self._make_layer(BasicBlock, 256, layers[1], stride=1, gcb_config=get_gcb_config(gcb_config, 1))

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        self.layer3 = self._make_layer(BasicBlock, 512, layers[2], stride=1, gcb_config=get_gcb_config(gcb_config, 2))

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)

        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=1, gcb_config=get_gcb_config(gcb_config, 3))

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)

    def init_weights(self):
        # Support both mmOCR 1.x style (no parameters) and original style (with pretrained parameter)
        if hasattr(self, 'init_cfg') and self.init_cfg is not None:
            # Use mmengine init system if init_cfg is provided
            from mmengine.model import initialize
            initialize(self, self.init_cfg)
        else:
            # Original initialization logic
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, gcb_config=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        
        # Create first block with downsample
        first_block = block(self.inplanes, planes, stride=stride, downsample=downsample)
        layers.append(first_block)
        
        self.inplanes = planes * block.expansion
        
        # Add remaining blocks
        for i in range(1, blocks):
            block_instance = block(self.inplanes, planes, stride=1, downsample=None)
            layers.append(block_instance)

        # Add ContextBlock after the layer if gcb_config is provided
        if gcb_config is not None:
            try:
                context_block = ContextBlock(
                    in_channels=planes * block.expansion,
                    ratio=gcb_config.get('ratio', 1.0/16),
                    pooling_type=gcb_config.get('pooling_type', 'att'),
                    fusion_types=(gcb_config.get('fusion_type', 'channel_add'),)
                )
                layers.append(context_block)
            except Exception:
                # Fallback: create a simple identity layer if ContextBlock fails
                print(f"Warning: Could not create ContextBlock with config {gcb_config}, skipping...")

        return nn.Sequential(*layers)

    def forward(self, x):
        f = []
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # (48, 160)

        x = self.maxpool1(x)
        x = self.layer1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        f.append(x)
        # (24, 80)

        x = self.maxpool2(x)
        x = self.layer2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        f.append(x)
        # (12, 40)

        x = self.maxpool3(x)

        x = self.layer3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.layer4(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        f.append(x)
        # (6, 40)

        return f

