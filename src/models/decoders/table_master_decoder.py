import copy
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from mmengine.model import ModuleList
from mmocr.registry import MODELS
from mmocr.models.common.dictionary import Dictionary
from mmocr.models.common.modules import PositionalEncoding
from mmocr.models.textrecog.decoders.base import BaseDecoder
from mmocr.models.textrecog.decoders.master_decoder import Embeddings

from structures.token_recog_data_sample import TokenRecogDataSample
from ..layers.tmd_layer import TMDLayer

@MODELS.register_module()
class TableMasterDecoder(BaseDecoder):
    """TableMaster Decoder module.
    
    Split to two transformer header at the last layer.
    Cls_layer is used to structure token classification.
    Bbox_layer is used to regress bbox coord.

    Args:
        n_layers (int): Number of attention layers. Defaults to 3.
        n_head (int): Number of parallel attention heads. Defaults to 8.
        d_model (int): Dimension of the input from previous model.
            Defaults to 512.
        decoder (dict, optional): Config dict for decoder layers. Should contain keys:
            'd_inner', 'attn_drop', 'ffn_drop'.
        module_loss (dict, optional): Config to build module_loss. Defaults
            to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        max_seq_len (int): Maximum output sequence length. Defaults to 30.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        n_layers: int = 3,
        n_head: int = 8,
        d_model: int = 512,
        decoder: Optional[Dict] = None,
        module_loss: Optional[Dict] = None,
        postprocessor: Optional[Dict] = None,
        dictionary: Optional[Union[Dict, Dictionary]] = None,
        max_seq_len: int = 30,
        init_cfg: Optional[Union[Dict, Sequence[Dict]]] = None,
    ):
        """
        Args:
            n_layers (int): Number of attention layers. Defaults to 3.
            n_head (int): Number of parallel attention heads. Defaults to 8.
            d_model (int): Dimension of the input from previous model.
                Defaults to 512.
            decoder (dict, optional): Config dict for decoder layers. Should contain keys:
                'd_inner', 'attn_drop', 'ffn_drop'.
        """
        decoder = decoder or {}
        d_inner = decoder.get('d_inner', 2048)
        attn_drop = decoder.get('attn_drop', 0.)
        ffn_drop = decoder.get('ffn_drop', 0.)

        super().__init__(
            module_loss=module_loss,
            postprocessor=postprocessor,
            dictionary=dictionary,
            init_cfg=init_cfg,
            max_seq_len=max_seq_len)
        
        decoder_layer = TMDLayer(
            d_model=d_model,
            n_head=n_head,
            d_inner=d_inner,
            attn_drop=attn_drop,
            ffn_drop=ffn_drop,
            operation_order=('norm', 'self_attn', 'norm', 'cross_attn', 'norm', 'ffn')
        )
        
        # Shared decoder layers
        self.decoder_layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(n_layers - 1)])
        # Separate classification and bbox layers
        self.cls_layer = ModuleList([copy.deepcopy(decoder_layer) for _ in range(1)])
        self.bbox_layer = ModuleList([copy.deepcopy(decoder_layer) for _ in range(1)])

        # Classification head
        self.cls_fc = nn.Linear(d_model, self.dictionary.num_classes)
        
        # Bbox regression head
        self.bbox_fc = nn.Sequential(
            nn.Linear(d_model, 4),
            nn.Sigmoid()
        )

        self.SOS = self.dictionary.start_idx
        self.PAD = self.dictionary.padding_idx
        self.max_seq_len = max_seq_len
        self.n_head = n_head

        self.embedding = Embeddings(d_model=d_model, vocab=self.dictionary.num_classes)

        self.positional_encoding = PositionalEncoding(d_hid=d_model, n_position=self.max_seq_len + 1)
        
        self.norm = nn.LayerNorm(d_model)
        self.softmax = nn.Softmax(dim=-1)

    def make_target_mask(self, tgt: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
        """Make target mask for self attention.

        Args:
            tgt (Tensor): Shape [N, l_tgt]
            device (torch.device): Mask device.

        Returns:
            Tensor: Mask of shape [N * self.n_head, l_tgt, l_tgt]
        """
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).bool()
        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device))
        tgt_mask = trg_pad_mask & trg_sub_mask

        # inverse for mmcv's BaseTransformerLayer
        tril_mask = tgt_mask.clone()
        tgt_mask = tgt_mask.float().masked_fill_(tril_mask == 0, -1e9)
        tgt_mask = tgt_mask.masked_fill_(tril_mask, 0)
        tgt_mask = tgt_mask.repeat(1, self.n_head, 1, 1)
        tgt_mask = tgt_mask.view(-1, tgt_len, tgt_len)
        return tgt_mask

    def decode(self, tgt_seq: torch.Tensor, feature: torch.Tensor,
               src_mask: torch.BoolTensor,
               tgt_mask: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode the input sequence.

        Args:
            tgt_seq (Tensor): Target sequence of shape: (N, T, C).
            feature (Tensor): Input feature map from encoder of
                shape: (N, C, H, W)
            src_mask (BoolTensor): The source mask of shape: (N, H*W).
            tgt_mask (BoolTensor): The target mask of shape: (N, T, T).

        Returns:
            Tuple[Tensor, Tensor]: The decoded classification and bbox outputs.
        """
        tgt_seq = self.embedding(tgt_seq)
        x = self.positional_encoding(tgt_seq)
        attn_masks = [tgt_mask, src_mask]
        
        # Shared decoder layers
        for layer in self.decoder_layers:
            x = layer(query=x, key=feature, value=feature, attn_masks=attn_masks)
        
        # Classification head
        cls_x = x
        for layer in self.cls_layer:
            cls_x = layer(query=cls_x, key=feature, value=feature, attn_masks=attn_masks)
        cls_x = self.norm(cls_x)
        cls_output = self.cls_fc(cls_x)
        
        # Bbox head
        bbox_x = x
        for layer in self.bbox_layer:
            bbox_x = layer(query=bbox_x, key=feature, value=feature, attn_masks=attn_masks)
        bbox_x = self.norm(bbox_x)
        bbox_output = self.bbox_fc(bbox_x)
        
        return cls_output, bbox_output

    def forward_train(self,
                      feat: Optional[torch.Tensor] = None,
                      out_enc: torch.Tensor = None,
                      data_samples: Sequence[TokenRecogDataSample] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward for training. Source mask will not be used here.

        Args:
            feat (Tensor, optional): Input feature map from backbone.
            out_enc (Tensor): Feature from encoder with positional encoding applied.
            data_samples (list[TokenRecogDataSample]): Batch of
                TokenRecogDataSample, containing gt_token and valid_ratio
                information.

        Returns:
            Tuple[Tensor, Tensor]: The raw classification and bbox logit tensors.
            Shape (N, T, C) where C is num_classes and (N, T, 4) for bbox.
        """
        # Use out_enc if provided (feature with positional encoding from encoder)
        # Otherwise use feat directly (similar to original mmOCR 0.x)
        if out_enc is not None:
            feature = out_enc
        else:
            feature = feat

        trg_seq = []
        for target in data_samples:
            trg_seq.append(target.gt_token.padded_indexes.to(feature.device))

        trg_seq = torch.stack(trg_seq, dim=0)

        src_mask = None
        tgt_mask = self.make_target_mask(trg_seq, device=feature.device)
        return self.decode(trg_seq, feature, src_mask, tgt_mask)

    def forward_test(self,
                     feat: Optional[torch.Tensor] = None,
                     out_enc: torch.Tensor = None,
                     data_samples: Sequence[TokenRecogDataSample] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward for testing.

        Args:
            feat (Tensor, optional): Input feature map from backbone.
            out_enc (Tensor): Feature from encoder with positional encoding applied.
            data_samples (list[TokenRecogDataSample]): Unused.

        Returns:
            Tuple[Tensor, Tensor]: Character probabilities and bbox outputs.
            Shape (N, self.max_seq_len, C) where C is num_classes
            and (N, self.max_seq_len, 4) for bbox.
        """
        # Use out_enc if provided (feature with positional encoding from encoder)
        # Otherwise use feat directly (similar to original mmOCR 0.x)
        if out_enc is not None:
            feature = out_enc
        else:
            feature = feat

        N = feature.shape[0]
        input = torch.full((N, 1),
                           self.SOS,
                           device=feature.device,
                           dtype=torch.long)
        cls_output = None
        bbox_output = None
        
        for _ in range(self.max_seq_len):
            target_mask = self.make_target_mask(input, device=feature.device)
            cls_out, bbox_out = self.decode(input, feature, None, target_mask)
            cls_output = cls_out
            bbox_output = bbox_out
            _, next_word = torch.max(cls_out, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        
        return self.softmax(cls_output), bbox_output