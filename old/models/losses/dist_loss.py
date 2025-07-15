import torch.nn as nn
from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class DistLoss(nn.Module):
    """Implementation of loss module for table master bbox regression branch
    with Distance loss.

    Args:
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
    """
    def __init__(self, reduction='none'):
        super().__init__()
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']
        self.dist_loss = self.build_loss(reduction)

    def build_loss(self, reduction, **kwargs):
        raise NotImplementedError

    def format(self, outputs, targets_dict):
        raise NotImplementedError

    def forward(self, outputs, targets_dict, img_metas=None):
        outputs, targets = self.format(outputs, targets_dict)
        loss_dist = self.dist_loss(outputs, targets.to(outputs.device))
        losses = dict(loss_dist=loss_dist)
        return losses