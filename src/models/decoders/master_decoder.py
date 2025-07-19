from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import ModuleList
from mmocr.registry import MODELS
from mmocr.models.common.dictionary import Dictionary
from mmocr.models.common.modules import PositionalEncoding
from mmocr.models.textrecog.decoders.base import BaseDecoder

from structures.table_master_data_sample import TableMasterDataSample
from ..layers.decoder_layer import DecoderLayer, Embeddings, clones

@MODELS.register_module()
class MasterDecoder(BaseDecoder):
    """Master Decoder module for text recognition.
    
    Standard transformer decoder for sequence-to-sequence text recognition.

    Args:
        n_layers (int): Number of attention layers. Defaults to 6.
        n_head (int): Number of parallel attention heads. Defaults to 8.
        d_model (int): Dimension of the input from previous model.
            Defaults to 512.
        decoder (dict, optional): Config dict for decoder layers. Should contain keys:
            'd_inner', 'attn_drop', 'ffn_drop'.
        text_loss (dict, optional): Config to build text_loss. Defaults
            to None.
        postprocessor (dict, optional): Config to build postprocessor.
            Defaults to None.
        max_seq_len (int): Maximum output sequence length. Defaults to 40.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        n_layers: int = 6,
        n_head: int = 8,
        d_model: int = 512,
        decoder: Optional[Dict] = None,
        text_loss: Optional[Dict] = None,
        postprocessor: Optional[Dict] = None,
        dictionary: Optional[Union[Dict, Dictionary]] = None,
        max_seq_len: int = 40,
        init_cfg: Optional[Union[Dict, Sequence[Dict]]] = None,
    ):
        # Set default decoder config if not provided
        if decoder is None:
            decoder = {
                'size': d_model,
                'self_attn': {
                    'headers': n_head,
                    'd_model': d_model,
                    'dropout': 0.1
                },
                'src_attn': {
                    'headers': n_head,
                    'd_model': d_model,
                    'dropout': 0.1
                },
                'feed_forward': {
                    'd_model': d_model,
                    'd_ff': d_model * 4,
                    'dropout': 0.1
                },
                'dropout': 0.1
            }

        super().__init__(
            postprocessor=postprocessor,
            dictionary=dictionary,
            init_cfg=init_cfg,
            max_seq_len=max_seq_len)

        if text_loss is not None:
            self.text_loss_module = MODELS.build(text_loss)

        # Create decoder layers using the same architecture as old version
        decoder_layer = DecoderLayer(**decoder)
        self.decoder_layers = clones(decoder_layer, n_layers)
        
        self.d_model = d_model
        
        # Classification head for text recognition
        self.cls_fc = nn.Linear(d_model, self.dictionary.num_classes)

        self.SOS = self.dictionary.start_idx
        self.PAD = self.dictionary.padding_idx
        self.max_seq_len = max_seq_len
        self.n_head = n_head

        self.embedding = Embeddings(d_model=d_model, vocab=self.dictionary.num_classes)
        self.positional_encoding = PositionalEncoding(d_hid=d_model, n_position=self.max_seq_len + 1)
        
        self.norm = nn.LayerNorm(d_model)

    def make_target_mask(self, tgt: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Make target mask for self attention.

        Args:
            tgt (Tensor): Shape [N, l_tgt]
            device (torch.device): Mask device.

        Returns:
            Tensor: Mask of shape [N, l_tgt, l_tgt]
        """
        # Create padding mask
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).bool()
        
        # Create subsequent mask (triangular mask)
        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device))
        
        # Combine masks
        tgt_mask = trg_pad_mask & trg_sub_mask
        
        return tgt_mask

    def decode(self, tgt_seq: torch.Tensor, feature: torch.Tensor,
               src_mask: torch.BoolTensor,
               tgt_mask: torch.BoolTensor) -> torch.Tensor:
        """Decode the input sequence.

        Args:
            tgt_seq (Tensor): Target sequence of shape: (N, T, C).
            feature (Tensor): Input feature map from encoder of
                shape: (N, C, H, W)
            src_mask (BoolTensor): The source mask of shape: (N, H*W).
            tgt_mask (BoolTensor): The target mask of shape: (N, T, T).

        Returns:
            Tensor: The decoded text classification outputs.
        """
        # Main process of transformer decoder
        x = self.embedding(tgt_seq)
        x = self.positional_encoding(x)
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(x, feature, src_mask, tgt_mask)
        
        x = self.norm(x)
        output = self.cls_fc(x)
        
        return output

    def forward_train(self,
                      feat: Optional[torch.Tensor] = None,
                      out_enc: torch.Tensor = None,
                      data_samples: Sequence[TableMasterDataSample] = None
                      ) -> torch.Tensor:
        """Forward for training. Source mask will not be used here.

        Args:
            feat (Tensor, optional): Input feature map from backbone.
            out_enc (Tensor): Feature from encoder with positional encoding applied.
            data_samples (list[TableMasterDataSample]): Batch of
                TableMasterDataSample, containing gt_tokens information.

        Returns:
            Tensor: The raw text classification logit tensor.
            Shape (N, T, C) where C is num_classes.
        """
        # Use out_enc if provided (feature with positional encoding from encoder)
        # Otherwise use feat directly (similar to original mmOCR 0.x)
        if out_enc is not None:
            feature = out_enc
        else:
            feature = feat

        trg_seq = []
        for target in data_samples:
            trg_seq.append(target.gt_tokens.clone())

        trg_seq = torch.stack(trg_seq, dim=0)

        src_mask = None
        tgt_mask = self.make_target_mask(trg_seq[:, :-1], device=feature.device)
        return self.decode(trg_seq[:, :-1], feature, src_mask, tgt_mask)

    def forward_test(self,
                     feat: Optional[torch.Tensor] = None,
                     out_enc: torch.Tensor = None,
                     data_samples: Sequence[TableMasterDataSample] = None
                     ) -> torch.Tensor:
        """Forward for testing.

        Args:
            feat (Tensor, optional): Input feature map from backbone.
            out_enc (Tensor): Feature from encoder with positional encoding applied.
            data_samples (list[TableMasterDataSample]): Unused.

        Returns:
            Tensor: Character probabilities.
            Shape (N, self.max_seq_len, C) where C is num_classes.
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
        
        # Greedy decoding similar to original MasterDecoder
        for _ in range(self.max_seq_len + 1):
            tgt_mask = self.make_target_mask(input, device=feature.device)
            output = self.decode(input, feature, None, tgt_mask)
            
            prob = F.softmax(output, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        
        return output
    
    def predict(self,
                feat: Optional[torch.Tensor] = None,
                out_enc: Optional[torch.Tensor] = None,
                data_samples: Optional[Sequence[TableMasterDataSample]] = None
                ) -> Sequence[TableMasterDataSample]:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            feat (Tensor, optional): Features from the backbone. Defaults
                to None.
            out_enc (Tensor, optional): Features from the encoder.
                Defaults to None.
            data_samples (list[TableMasterDataSample], optional): A list of
                N datasamples, containing meta information and gold
                annotations for each of the images. Defaults to None.

        Returns:
            list[TableMasterDataSample]: A list of N datasamples of prediction results.
        """
        out_logits = self.forward_test(feat=feat, out_enc=out_enc,
                                       data_samples=data_samples)
        
        if self.postprocessor is None:
            return out_logits
        
        return self.postprocessor(
            out_logits, data_samples=data_samples)
    
    def loss(self,
             feat: Optional[torch.Tensor] = None,
             out_enc: Optional[torch.Tensor] = None,
             data_samples: Optional[Sequence[TableMasterDataSample]] = None
             ) -> Dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feat (Tensor, optional): Features from the backbone. Defaults
                to None.
            out_enc (Tensor, optional): Features from the encoder.
                Defaults to None.
            data_samples (list[TableMasterDataSample], optional): A list of
                N datasamples, containing meta information and gold
                annotations for each of the images. Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        out_logits = self.forward_train(feat=feat, out_enc=out_enc,
                                        data_samples=data_samples)
        
        loss_dict = {}
        if hasattr(self, 'text_loss_module'):
            loss_dict.update(
                self.text_loss_module(out_logits, data_samples))
        
        return loss_dict
