# Copyright (c) Lê Hoàn Chân. All rights reserved.
import math
from typing import Dict, List, Optional, Sequence, Union
import torch
import torch.nn as nn
from mmocr.registry import MODELS
from structures.table_master_data_sample import TableMasterDataSample
from mmocr.models.textrecog.encoders.base import BaseEncoder

@MODELS.register_module()
class PositionalEncoding(BaseEncoder):
    """Implement the PE function.
    
    Args:
        d_model (int): The dimension of the model. Defaults to 512.
        dropout (float): Dropout probability. Defaults to 0.0.
        max_len (int): Maximum sequence length. Defaults to 5000.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self, 
                 d_model: int = 512, 
                 dropout: float = 0., 
                 max_len: int = 5000,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, 
                feat: torch.Tensor,
                data_samples: Optional[Sequence[TableMasterDataSample]] = None,
                **kwargs) -> torch.Tensor:
        """Forward function.
        
        Args:
            feat (torch.Tensor): Input feature tensor.
            data_samples (Sequence[TableMasterDataSample], optional): Batch of
                TableMasterDataSample. Defaults to None.
                
        Returns:
            torch.Tensor: Output tensor with positional encoding added.
        """
        if len(feat.shape) > 3:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h*w) # flatten 2D feature map
            feat = feat.permute((0,2,1))
        feat = feat + self.pe[:, :feat.size(1)] # pe 1*5000*512
        return self.dropout(feat)