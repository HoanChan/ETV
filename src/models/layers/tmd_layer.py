import torch.nn as nn
from mmcv.cnn.bricks.transformer import BaseTransformerLayer

class TMDLayer(nn.Module):
    def __init__(self, d_model, n_head, d_inner, attn_drop, ffn_drop, operation_order=None):
        super().__init__()
        if operation_order is None:
            operation_order = ('norm', 'self_attn', 'norm', 'cross_attn', 'norm', 'ffn')
        self.layer = BaseTransformerLayer(
            operation_order=operation_order,
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=d_model,
                num_heads=n_head,
                attn_drop=attn_drop,
                dropout_layer=dict(type='Dropout', drop_prob=attn_drop),
            ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=d_model,
                feedforward_channels=d_inner,
                ffn_drop=ffn_drop,
                dropout_layer=dict(type='Dropout', drop_prob=ffn_drop),
            ),
            norm_cfg=dict(type='LN'),
            batch_first=True,
        )

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)