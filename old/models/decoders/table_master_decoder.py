import torch
import torch.nn as nn
import torch.nn.functional as F
from mmocr.models.builder import DECODERS
from mmocr.models.textrecog.decoders import BaseDecoder
from ..encoders.positional_encoding import PositionalEncoding
from .modules import Embeddings, DecoderLayer, clones

@DECODERS.register_module()
class TableMasterDecoder(BaseDecoder):
    """
    Split to two transformer header at the last layer.
    Cls_layer is used to structure token classification.
    Bbox_layer is used to regress bbox coord.
    """
    def __init__(self,
                 N,
                 decoder,
                 d_model,
                 num_classes,
                 start_idx,
                 padding_idx,
                 max_seq_len,
                 ):
        super(TableMasterDecoder, self).__init__()
        self.layers = clones(DecoderLayer(**decoder), N-1)
        self.cls_layer = clones(DecoderLayer(**decoder), 1)
        self.bbox_layer = clones(DecoderLayer(**decoder), 1)
        self.cls_fc = nn.Linear(d_model, num_classes)
        self.bbox_fc = nn.Sequential(
            nn.Linear(d_model, 4),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(decoder.size)
        self.embedding = Embeddings(d_model=d_model, vocab=num_classes)
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        self.SOS = start_idx
        self.PAD = padding_idx
        self.max_length = max_seq_len

    def make_mask(self, src, tgt):
        """
        Make mask for self attention.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).byte()

        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=src.device))

        tgt_mask = trg_pad_mask & trg_sub_mask
        return None, tgt_mask

    def decode(self, input, feature, src_mask, tgt_mask):
        # main process of transformer decoder.
        x = self.embedding(input)
        x = self.positional_encoding(x)

        # origin transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, feature, src_mask, tgt_mask)

        # cls head
        for layer in self.cls_layer:
            cls_x = layer(x, feature, src_mask, tgt_mask)
        cls_x = self.norm(cls_x)

        # bbox head
        for layer in self.bbox_layer:
            bbox_x = layer(x, feature, src_mask, tgt_mask)
        bbox_x = self.norm(bbox_x)

        return self.cls_fc(cls_x), self.bbox_fc(bbox_x)

    def greedy_forward(self, SOS, feature, mask):
        input = SOS
        output = None
        for i in range(self.max_length+1):
            _, target_mask = self.make_mask(feature, input)
            out, bbox_output = self.decode(input, feature, None, target_mask)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        return output, bbox_output

    def forward_train(self, feat, out_enc, targets_dict, img_metas=None):
        # x is token of label
        # feat is feature after backbone before pe.
        # out_enc is feature after pe.
        device = feat.device
        if isinstance(targets_dict, dict):
            padded_targets = targets_dict['padded_targets'].to(device)
        else:
            padded_targets = targets_dict.to(device)

        src_mask = None
        _, tgt_mask = self.make_mask(out_enc, padded_targets[:,:-1])
        return self.decode(padded_targets[:, :-1], out_enc, src_mask, tgt_mask)

    def forward_test(self, feat, out_enc, img_metas):
        src_mask = None
        batch_size = out_enc.shape[0]
        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        output, bbox_output = self.greedy_forward(SOS, out_enc, src_mask)
        return output, bbox_output

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        return self.forward_test(feat, out_enc, img_metas)